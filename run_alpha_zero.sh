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
module load python/3.13


# Setup Python environments
cd $SLURM_TMPDIR
python -m venv pyenv
. pyenv/bin/activate
git clone ~/node_arena
cd node_arena
pip install jax jaxlib numpy flax chex orbax-checkpoint optax h5py docopt --no-index
cmake -S . -B build
cmake --build build


export PYTHONPATH=${SLURM_TMPDIR}/node_arena
export XLA_PYTHON_CLIENT_MEMORY_PREALLOC=false

GAME=$1
SIZE=$2
BASE_CONFIG="${GAME}_${SIZE//,/_}"
CONFIG_TEMPLATE_DIR=$3
SEED=$SLURM_ARRAY_TASK_ID

OUTPUT_DIR="${HOME}/scratch/alpha_zero/${GAME}_${BASE_CONFIG}/seed_$SEED"

python python/run_alpha_zero.py $GAME $SIZE $OUTPUT_DIR --base-config=$CONFIG_TEMPLATE_DIR/${BASE_CONFIG}_train.yaml --base-eval-cfg=$CONFIG_TEMPLATE_DIR/${BASE_CONFIG}_eval.yaml --seed=$SEED
