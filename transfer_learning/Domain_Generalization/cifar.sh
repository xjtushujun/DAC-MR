
# CIFAR-C
SEEDS="0 1 2"
SHARED_ARGS="\
    --dataset cifar-c \
    --num_epochs 100 \
    --n_samples_per_group 2000 \
    --test_n_samples_per_group 3000 \
    --eval_on val test \
    --seeds ${SEEDS} \
    --meta_batch_size 3 \
    --support_size 100 \
    --epochs_per_eval 1 \
    --optimizer sgd \
    --learning_rate 1e-2 \
    --weight_decay 1e-4 \
    --log_wandb 0"

N_CONTEXT_CHANNELS=3 # For CML

# python run.py --exp_name arm-cml --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels $N_CONTEXT_CHANNELS --adapt_bn 1  $SHARED_ARGS
# python run.py --exp_name arm-ll --algorithm ARM-LL --sampler group --uniform_over_groups 1  $SHARED_ARGS
python run.py --exp_name arm-bn --algorithm ARM-BN --sampler group --uniform_over_groups 1  $SHARED_ARGS



