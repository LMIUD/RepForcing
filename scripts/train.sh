#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"
PYTHON=${PYTHON:-python}

if [ "${1:-}" = "--list" ]; then
    exec "$PYTHON" -c 'from configs.registry import available; print("\n".join(available()))'
fi
VARIANT=${1:?Usage: sh scripts/train.sh <rft.framewise|rft.chunkwise|dmd.framewise|dmd.chunkwise> [arguments]}
shift
case "$VARIANT" in
    rft.framewise|rft.chunkwise|dmd.framewise|dmd.chunkwise) ;;
    *) printf 'Unknown configuration: %s\n' "$VARIANT" >&2; exit 2 ;;
esac

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
LOGDIR=${LOGDIR:-logs/$(printf '%s' "$VARIANT" | tr '.' '_')}
export PYTHONUNBUFFERED=1

exec "$PYTHON" -m torch.distributed.run \
    --nnodes "$NNODES" \
    --nproc_per_node "$GPUS_PER_NODE" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "train.py" --config "$VARIANT" --logdir "$LOGDIR" \
    --wandb-save-dir "$LOGDIR" "$@"
