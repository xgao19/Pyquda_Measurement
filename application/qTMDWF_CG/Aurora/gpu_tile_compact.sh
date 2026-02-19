#!/bin/bash
set -euo pipefail

lr=${PALS_LOCAL_RANKID}
case $lr in
  0) gpu=0 ;;
  1) gpu=1 ;;
  2) gpu=2 ;;
  3) gpu=3 ;;
  4) gpu=4 ;;
  5) gpu=5 ;;
  6) gpu=0 ;;   # share GPU0
  7) gpu=1 ;;   # share GPU1
  *) gpu=$((lr % 6)) ;;
esac

unset EnableWalkerPartition
export EnableImplicitScaling=0
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=${gpu}
export ONEAPI_DEVICE_FILTER=gpu,level_zero

echo "rank $PALS_RANKID ; local rank $lr ; ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK" >&2
exec "$@"

