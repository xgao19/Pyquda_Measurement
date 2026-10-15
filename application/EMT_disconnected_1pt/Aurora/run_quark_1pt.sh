#!/bin/bash
set -euo pipefail

# Benchmark settings; run this script inside an existing PBS allocation.
configuration=1050
n_base_noise=1
flow_batch_size=1
mpi_geometry="2.2.4.4"
mg_block="4.4.4.4;4.4.2.2"

benchmark_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
measurement_root="$(cd "$benchmark_dir/../../.." && pwd)"

if [[ -z "${PBS_NODEFILE:-}" || ! -f "$PBS_NODEFILE" ]]; then
  echo "ERROR: run inside an 8-node PBS allocation" >&2
  exit 2
fi
mapfile -t hosts < <(awk '!seen[$0]++' "$PBS_NODEFILE")
if [[ "${#hosts[@]}" -lt 8 ]]; then
  echo "ERROR: expected at least 8 unique hosts, got ${#hosts[@]}" >&2
  exit 2
fi
mkdir -p "$benchmark_dir/scripts" "$benchmark_dir/data" "$benchmark_dir/log"
printf "%s\n" "${hosts[@]:0:8}" > "$benchmark_dir/scripts/hosts"

source /lus/flare/projects/StructNGB/xgao/software_gradientflow_261009/activate-pyquda-develop.sh
# Use the corrected nested HP and spin-color dilution without changing other jobs.
export PYQUDA_ROOT="$measurement_root"
export PYTHONPATH="$PYQUDA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_P2P=0
export PYQUDA_MEASUREMENT_TIMERS=1
unset QUDA_RESOURCE_PATH
unset QUDA_ENABLE_MPS

/opt/cray/pals/1.8/bin/mpiexec \
  -n 64 -ppn 8 --hostfile "$benchmark_dir/scripts/hosts" -envall \
  --cpu-bind=depth "$benchmark_dir/tile_flat_affinity.sh" \
  python -u "$benchmark_dir/check_tile_device.py" \
  > "$benchmark_dir/log/tile_precheck.o" \
  2> "$benchmark_dir/log/tile_precheck.e"

/usr/bin/time -p /opt/cray/pals/1.8/bin/mpiexec \
  -n 64 -ppn 8 --hostfile "$benchmark_dir/scripts/hosts" -envall \
  --cpu-bind=depth "$benchmark_dir/tile_flat_affinity.sh" \
  python -u "$benchmark_dir/Pyquda_EMT_disconnected_quark_1pt.py" \
    --config_num "$configuration" \
    --mpi_geometry "$mpi_geometry" \
    --mg-block "$mg_block" \
    --n-base-noise "$n_base_noise" \
    --flow-batch-size "$flow_batch_size" \
    --data-dir "$benchmark_dir/data" \
    --resource-path "$benchmark_dir/cache" \
  > "$benchmark_dir/log/strange_cfg${configuration}.o" \
  2> "$benchmark_dir/log/strange_cfg${configuration}.e"

python -u "$benchmark_dir/validate_strange.py" \
  --configuration "$configuration" --n-base-noise "$n_base_noise" \
  > "$benchmark_dir/log/validation.o" \
  2> "$benchmark_dir/log/validation.e"
