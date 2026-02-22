"""Low-rank linear layer for parameter-efficient pretraining.

Supports both static low-rank and timestep-dependent (dynamic) low-rank
modes.  When timestep-dependent mode is active, the number of active rank
slices is determined per-sample by a logistic schedule over the
diffusion timestep t ∈ [0, 1].

The default schedule maps low t (clean) → high rank, high t (noisy) → low
rank.  Setting ``_reverse_schedule = True`` reverses this so that high t
(noisy) → high rank and low t (clean) → low rank.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankLinear(nn.Module):
  """A linear layer factorised into two low-rank matrices.

  Replaces a dense ``nn.Linear(in_features, out_features)`` with
  ``y = x @ B^T @ A^T + bias`` where ``A`` has shape
  ``(out_features, rank)`` and ``B`` has shape ``(rank, in_features)``.

  The rank is determined so that the total parameter count of ``A``
  and ``B`` equals a target percentage of the original dense layer's
  parameter count (excluding bias):

      rank = floor(pct * in_features * out_features
                   / (in_features + out_features))

  Timestep-dependent mode
  -----------------------
  When ``_timestep`` is set (injected externally), the layer computes
  per-sample active ranks via a logistic increasing schedule over the
  diffusion timestep t ∈ [0, 1] and masks the intermediate rank
  dimension accordingly.

  Args:
    in_features: Size of each input sample.
    out_features: Size of each output sample.
    rank_percentage: Fraction (0, 1] of the original dense layer's
        parameter count to retain.
    bias: If ``True``, adds a learnable bias of shape
        ``(out_features,)``.
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      rank_percentage: float = 1.0,
      bias: bool = False,
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.rank_percentage = rank_percentage

    # Compute rank from the target parameter-count percentage.
    dense_params = in_features * out_features
    rank = int(
        math.floor(
            rank_percentage * dense_params
            / (in_features + out_features)
        )
    )
    rank = max(1, min(rank, min(in_features, out_features)))
    self.rank = rank

    # Factorised weight: W ≈ A @ B  (out×r) @ (r×in)
    self.A = nn.Parameter(torch.empty(out_features, rank))
    self.B = nn.Parameter(torch.empty(rank, in_features))

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_features))
    else:
      self.register_parameter('bias', None)

    # ----- timestep-dependent rank gating (set externally) -----
    self._timestep: torch.Tensor | None = None   # t ∈ [0, 1]; t=1 noisy, t≈0 clean
    self._r_min_ratio: float = 0.4
    self._logistic_k: float = 8.0
    self._logistic_m: float = 0.6
    self._reverse_schedule: bool = False  # if True, high noise → high rank

    self.reset_parameters()

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
    if self.bias is not None:
      fan_in = self.in_features
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      nn.init.uniform_(self.bias, -bound, bound)

  @staticmethod
  def compute_adjusted_rank_percentage(
      in_features: int,
      out_features: int,
      rank_percentage: float,
      r_min_ratio: float,
      logistic_k: float,
      logistic_m: float,
      sampling_eps: float,
      mode: str = "match_high_rank",
      reverse_schedule: bool = False,
  ) -> float:
    """Compute adjusted rank_percentage for timestep-dependent init modes.

    Args:
      in_features: Input dimension of the layer.
      out_features: Output dimension of the layer.
      rank_percentage: Target rank percentage (for static low-rank).
      r_min_ratio: Minimum rank ratio for timestep gating.
      logistic_k: Logistic schedule steepness.
      logistic_m: Logistic schedule midpoint.
      sampling_eps: Lower bound for timestep sampling.
      mode: Initialization mode, either 'match_high_rank' or
          'match_expected_rank'.
      reverse_schedule: If True, reverse the schedule direction
          (high noise → high rank).

    Returns:
      Adjusted rank_percentage to use when creating LowRankLinear layers.
          - 'match_high_rank': returns rank_percentage unchanged
          - 'match_expected_rank': returns adjusted percentage so that
            expected activated rank matches static low-rank
    """
    if mode == "match_high_rank":
      return rank_percentage

    if mode != "match_expected_rank":
      raise ValueError(
          f"Unknown mode '{mode}'. Expected 'match_high_rank' or "
          "'match_expected_rank'.")

    # Create temporary layer with original percentage to get target rank
    temp_layer = LowRankLinear(in_features, out_features, rank_percentage)
    target_rank = temp_layer.rank  # This is the static low-rank rank

    # Set schedule parameters
    temp_layer._r_min_ratio = r_min_ratio
    temp_layer._logistic_k = logistic_k
    temp_layer._logistic_m = logistic_m
    temp_layer._reverse_schedule = reverse_schedule

    # Calculate expected activation percentage
    _, expected_pct = temp_layer.calculate_expected_rank(sampling_eps)
    expected_ratio = expected_pct / 100.0

    # Adjusted percentage = original / expected_ratio
    # This gives us a higher full rank, so that expected_rank ≈ target_rank
    adjusted_percentage = rank_percentage / expected_ratio

    # Clamp to valid range [0, 1]
    adjusted_percentage = min(adjusted_percentage, 1.0)

    return adjusted_percentage

  # ------------------------------------------------------------------
  # Logistic schedule:  rank varies with diffusion timestep
  # ------------------------------------------------------------------
  def _active_ranks(
      self,
      t: torch.Tensor,
  ) -> torch.Tensor:
    """Map per-sample timestep t → active rank.

    Default schedule (*logistic increasing*): low t (clean) → high rank,
    high t (noisy) → low rank.

    Reversed schedule (``_reverse_schedule=True``): high t (noisy) →
    high rank, low t (clean) → low rank.

    Args:
      t: [B] float tensor, diffusion timestep in [0, 1].
         t=1 → fully noisy, t≈0 → clean.

    Returns:
      [B] long tensor of per-sample active ranks.
    """
    r = self.rank
    r_min = max(1, int(round(self._r_min_ratio * r)))
    k = self._logistic_k
    m = self._logistic_m

    t = t.to(self.A.device)

    if self._reverse_schedule:
      # Reversed: frac=0 → t=0 (clean), frac=1 → t=1 (noisy)
      # ⇒ high noise → high rank, low noise → low rank
      frac = t.clamp(0, 1)
    else:
      # Default: frac=0 → t=1 (noisy), frac=1 → t=0 (clean)
      # ⇒ low noise → high rank, high noise → low rank
      frac = (1.0 - t).clamp(0, 1)

    # Normalised sigmoid so that f(0)=0, f(1)=1
    raw_s = torch.sigmoid(torch.tensor(k, device=t.device, dtype=t.dtype) * (frac - m))
    s_min = torch.sigmoid(torch.tensor(k * (0.0 - m), device=t.device, dtype=t.dtype))
    s_max = torch.sigmoid(torch.tensor(k * (1.0 - m), device=t.device, dtype=t.dtype))
    s = (raw_s - s_min) / (s_max - s_min + 1e-8)

    r_t = (r_min + (r - r_min) * s).floor().clamp(min=r_min, max=r)
    return r_t.to(torch.long)

  def calculate_expected_rank(self, sampling_eps: float = 1e-3,
                              n_samples: int = 10000) -> tuple[float, float]:
    """Calculate expected active rank over the timestep distribution.

    Samples timesteps uniformly from [sampling_eps, 1] (matching the
    training distribution), computes active ranks, and returns the mean.

    Args:
      sampling_eps: Lower bound for timestep sampling (default 1e-3).
      n_samples: Number of timestep samples for Monte Carlo estimate.

    Returns:
      (expected_rank, expected_percentage): Expected active rank and
          the percentage relative to full rank.
    """
    # Sample timesteps uniformly from [sampling_eps, 1]
    t_samples = torch.rand(n_samples) * (1 - sampling_eps) + sampling_eps
    
    # Compute active ranks for each timestep
    ranks = self._active_ranks(t_samples)
    
    # Calculate expected rank
    expected_rank = ranks.float().mean().item()
    expected_percentage = (expected_rank / self.rank) * 100
    
    return expected_rank, expected_percentage

  # ------------------------------------------------------------------
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # ---- static path (no timestep conditioning) --------------------
    if self._timestep is None:
      # x @ B^T  → (..., rank), then  @ A^T → (..., out_features)
      return F.linear(F.linear(x, self.B), self.A, self.bias)

    # ---- timestep-dependent path -----------------------------------
    t = self._timestep             # [B]  t ∈ [0, 1]

    B_x = x.shape[0]
    B_t = t.shape[0]

    # Expand t if batch dims differ (e.g. seq-level flattening)
    if B_x == B_t:
      t_expanded = t
    elif B_x % B_t == 0:
      repeat = B_x // B_t
      t_expanded = t.unsqueeze(1).expand(-1, repeat).contiguous().view(-1)
    else:
      repeat = math.ceil(B_x / B_t)
      t_expanded = t.unsqueeze(1).expand(-1, repeat).contiguous().view(-1)[:B_x]

    r_t = self._active_ranks(t_expanded)  # [B_x]
    r = self.rank

    # Fast path: if the whole batch shares the same active rank, slice
    if bool(torch.all(r_t == r_t[0])):
      r_active = int(r_t[0].item())
      Bx = F.linear(x, self.B[:r_active, :])            # (..., r_active)
      return F.linear(Bx, self.A[:, :r_active], self.bias)

    # General path: mixed ranks in batch → mask activations
    Bx = F.linear(x, self.B)                              # (..., r)
    idx = torch.arange(r, device=x.device)
    mask = (idx.unsqueeze(0) < r_t.unsqueeze(1))           # [B_x, r]

    # Handle 3-D tensors (batch, seq, rank)
    if Bx.dim() == 3:
      mask = mask.unsqueeze(1)                             # [B_x, 1, r]

    Bx = Bx * mask
    return F.linear(Bx, self.A, self.bias)

  def extra_repr(self) -> str:
    return (
        f'in_features={self.in_features}, '
        f'out_features={self.out_features}, '
        f'rank={self.rank}, '
        f'rank_percentage={self.rank_percentage}, '
        f'bias={self.bias is not None}'
    )
