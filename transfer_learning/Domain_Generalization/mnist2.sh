
# MNIST
SEEDS="0 1 2"
SHARED_ARGS="\
    --dataset mnist \
    --num_epochs 200 \
    --eval_on val test \
    --n_samples_per_group 300 \
    --seeds ${SEEDS} \
    --meta_batch_size 6 \
    --epochs_per_eval 10 \
    --optimizer adam \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --log_wandb 0"


N_CONTEXT_CHANNELS=12 # For CML

# python run.py --exp_name arm-cml --algorithm ARM-CML --sampler group --uniform_over_groups 1 --n_context_channels $N_CONTEXT_CHANNELS  $SHARED_ARGS
python run.py --exp_name arm-ll --algorithm ARM-LL --sampler group --uniform_over_groups 1  $SHARED_ARGS
# python run.py --exp_name arm-bn --algorithm ARM-BN --sampler group --uniform_over_groups 1  $SHARED_ARGS

