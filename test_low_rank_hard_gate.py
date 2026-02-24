import torch

from tabdiff.modules.low_rank import LowRankLinear


def _make_layer(**kwargs):
    layer = LowRankLinear(
        in_features=8,
        out_features=6,
        rank_percentage=0.75,
        bias=True,
        mode='learnable',
        **kwargs,
    )
    return layer


def test_gradient_path_hard_ste():
    layer = _make_layer(
        learnable_gate_mode='hard_ste',
        learnable_gate_threshold=0.5,
        learnable_min_active_rank=1,
    )
    layer.train()
    layer._timestep = torch.rand(4)
    x = torch.randn(4, 8)
    loss = layer(x).pow(2).mean()
    loss.backward()

    assert layer.schedule is not None
    assert layer.schedule.alpha_raw.grad is not None
    assert layer.schedule.rho_raw.grad is not None
    assert torch.isfinite(layer.schedule.alpha_raw.grad).all()
    assert torch.isfinite(layer.schedule.rho_raw.grad).all()


def test_hard_threshold_forward_values():
    tau = 0.5
    layer = _make_layer(
        learnable_gate_mode='hard_ste',
        learnable_gate_threshold=tau,
        learnable_min_active_rank=0,
    )
    t = torch.tensor([0.3, 0.7], dtype=torch.float32)
    gate, hard_mask, _ = layer._build_learnable_gate(t)
    p = layer.schedule.soft_mask(t)

    expected_hard = p > tau
    expected_gate_hard = p * expected_hard.to(p.dtype)

    assert torch.equal(hard_mask, expected_hard)
    assert torch.allclose(gate, expected_gate_hard, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(gate[~expected_hard]) == 0


def test_min_active_rank_behavior():
    layer_keep_one = _make_layer(
        learnable_gate_mode='hard_ste',
        learnable_gate_threshold=1.1,
        learnable_min_active_rank=1,
    )
    t = torch.tensor([0.2, 0.8], dtype=torch.float32)
    _, _, hard_counts_keep = layer_keep_one._build_learnable_gate(t)
    assert torch.all(hard_counts_keep >= 1)

    layer_allow_zero = _make_layer(
        learnable_gate_mode='hard_ste',
        learnable_gate_threshold=1.1,
        learnable_min_active_rank=0,
    )
    _, _, hard_counts_zero = layer_allow_zero._build_learnable_gate(t)
    assert torch.all(hard_counts_zero == 0)


def test_eval_slice_equivalence():
    layer = _make_layer(
        learnable_gate_mode='hard_ste',
        learnable_gate_threshold=0.5,
        learnable_min_active_rank=1,
        learnable_eval_slice=True,
    )
    layer.eval()

    x = torch.randn(5, 8)
    layer._timestep = torch.full((5,), 0.6, dtype=torch.float32)
    out_fast = layer(x)

    layer.learnable_eval_slice = False
    out_full = layer(x)

    assert torch.allclose(out_fast, out_full, atol=1e-6, rtol=1e-6)


def test_backward_compatible_defaults():
    layer = LowRankLinear(
        in_features=8,
        out_features=6,
        rank_percentage=0.75,
        bias=True,
        mode='learnable',
    )
    layer.eval()
    layer._timestep = torch.rand(3)
    x = torch.randn(3, 8)
    out = layer(x)

    assert layer.learnable_gate_mode == 'soft'
    assert abs(layer.learnable_gate_threshold - 0.5) < 1e-12
    assert layer.learnable_min_active_rank == 1
    assert layer.learnable_eval_slice is True
    assert out.shape == (3, 6)


def test_optimizer_step_smoke():
    layer = _make_layer(
        learnable_gate_mode='hard_ste',
        learnable_gate_threshold=0.5,
        learnable_min_active_rank=1,
    )
    layer.train()
    layer._timestep = torch.rand(4)
    x = torch.randn(4, 8)

    opt = torch.optim.SGD(layer.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss = layer(x).pow(2).mean()
    loss.backward()
    opt.step()

    assert torch.isfinite(loss).item()


if __name__ == '__main__':
    test_gradient_path_hard_ste()
    test_hard_threshold_forward_values()
    test_min_active_rank_behavior()
    test_eval_slice_equivalence()
    test_backward_compatible_defaults()
    test_optimizer_step_smoke()
    print('All low-rank hard-gate tests passed.')
