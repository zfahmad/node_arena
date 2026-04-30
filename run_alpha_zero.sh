#!/bin/bash
#SBATCH --array=0-10
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00


# SOCKS5 proxy
if [ "$SLURM_TMPDIR" != "" ]; then
    echo "Setting up SOCKS5 proxy..."
    ssh -q -N -T -f -D 8888 `echo $SSH_CONNECTION | cut -d " " -f 3`
    export ALL_PROXY=socks5h://localhost:8888
fi


# Setup modules
module load python/3.13 cuda


# Setup Python environments
cd $SLURM_TMPDIR
python -m venv pyenv
. pyenv/bin/activate
git clone ~/node_arena
cd node_arena
pip install jax_cuda12_pjrt jaxlib numpy flax chex orbax-checkpoint optax h5py docopt --no-index


export PYTHONPATH=${SLURM_TMPDIR}/node_arena
export XLA_PYTHON_CLIENT_MEMORY_PREALLOC=false


SEEDS=(0..10)
CONFIG_PATH="${HOME}/scratch/alpha_zero/connect_four/seed_${SEEDS[$SLURM_ARRAY_TASK_ID]}/alpha_zero.yaml"
OUTPUT_DIR="${HOME}/scratch/alpha_zero_connect_four/seed_${SEEDS[$SLURM_ARRAY_TASK_ID]}"


python python/train_alpha_zero.py $CONFIG_PATH --output=$OUTPUT_DIR
