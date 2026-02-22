# EXP_NAME="tabdiff_full_rank"
# DATASET="adult"

# python main.py --dataname $DATASET --mode test --report --no_wandb --exp_name "$EXP_NAME"




python main.py --dataname adult --mode test --report --no_wandb \
    --low_rank_mode dynamic --rank_percentage 0.7 --dynamic_rank_init match_high_rank \
    --exp_name tabdiff_match_expected_rank


# python main.py --dataname adult --mode test --report --no_wandb \
#     --low_rank_mode static --rank_percentage 0.5 \
#     --exp_name tabdiff_slr