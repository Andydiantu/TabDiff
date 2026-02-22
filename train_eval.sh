


# python main.py --dataname adult --mode train \
#     --exp_name tabdiff_full_rank 


# python main.py --dataname adult --mode test --report --no_wandb \
#     --exp_name tabdiff_full_rank

# echo "TabDiff Full Rank Done"


# python main.py --dataname adult --mode train --low_rank_mode dynamic --rank_percentage 0.5 \
#     --dynamic_rank_init match_expected_rank --exp_name tabdiff_match_expected_rank


# python main.py --dataname adult --mode test --report --no_wandb \
#     --low_rank_mode dynamic --rank_percentage 0.5 --dynamic_rank_init match_expected_rank \
#     --exp_name tabdiff_match_expected_rank


# python main.py --dataname adult --mode train --low_rank_mode dynamic --rank_percentage 0.5 \
#     --dynamic_rank_init match_high_rank --exp_name tabdiff_match_high_rank


python main.py --dataname adult --mode test --report --no_wandb \
    --low_rank_mode dynamic --rank_percentage 0.5 --dynamic_rank_init match_high_rank \
    --exp_name tabdiff_match_high_rank

echo "TabDiff Match High Rank Done"


# python main.py --dataname adult --mode train --low_rank_mode static --rank_percentage 0.5 \
#     --exp_name tabdiff_slr

# python main.py --dataname adult --mode test --report --no_wandb \
#     --low_rank_mode static --rank_percentage 0.5 \
#     --exp_name tabdiff_slr