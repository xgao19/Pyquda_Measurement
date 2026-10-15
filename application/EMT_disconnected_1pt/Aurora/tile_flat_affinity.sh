#!/bin/bash
set -euo pipefail

local_rank="${PALS_LOCAL_RANKID:-${MPI_LOCALRANKID:-${PMI_LOCAL_RANK:-0}}}"
tile=$((local_rank % 12))

export NEOReadDebugKeys=1
export EnableImplicitScaling=0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="$tile"
unset ONEAPI_DEVICE_FILTER

exec "$@"

