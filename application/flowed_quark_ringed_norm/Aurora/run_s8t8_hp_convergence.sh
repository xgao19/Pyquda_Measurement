#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
measurement_root="$(cd "$script_dir/../../.." && pwd)"

cd "$script_dir"
source /lus/flare/projects/StructNGB/xgao/software_gradientflow/activate-pyquda-develop.sh

unset ZE_FLAT_DEVICE_HIERARCHY
unset ZE_AFFINITY_MASK
unset ONEAPI_DEVICE_SELECTOR
unset ONEAPI_DEVICE_FILTER

export QUDA_ENABLE_P2P="${QUDA_ENABLE_P2P:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

export FLOWED_RINGED_NRANKS="${FLOWED_RINGED_NRANKS:-2}"
export FLOWED_RINGED_MPI_GEOMETRY="${FLOWED_RINGED_MPI_GEOMETRY:-1.1.1.2}"
export FLOWED_RINGED_CONFIG_NUM="${FLOWED_RINGED_CONFIG_NUM:-0}"
export FLOWED_RINGED_LAT_TAG="${FLOWED_RINGED_LAT_TAG:-S8T8}"
export FLOWED_RINGED_GAUGE_PATH="${FLOWED_RINGED_GAUGE_PATH:-$measurement_root/test_gauge/S8T8_wilson_b6.0}"
export FLOWED_RINGED_FLOW_STEPS="${FLOWED_RINGED_FLOW_STEPS:-1}"
export FLOWED_RINGED_FLOW_TYPE="${FLOWED_RINGED_FLOW_TYPE:-wilson}"
export FLOWED_RINGED_FLOW_EPSILON="${FLOWED_RINGED_FLOW_EPSILON:-0.207936}"
export FLOWED_RINGED_N_ZN="${FLOWED_RINGED_N_ZN:-2}"
export FLOWED_RINGED_MAXITER="${FLOWED_RINGED_MAXITER:-300}"
export FLOWED_RINGED_TOL="${FLOWED_RINGED_TOL:-1e-10}"

bench_root="${FLOWED_RINGED_BENCH_ROOT:-$script_dir/benchmark/s8t8_hp_convergence}"
mkdir -p "$bench_root/log" "$bench_root/data" "$bench_root/cache"

run_case() {
  local case_name="$1"
  local noise_scheme="$2"
  local n_vec="$3"
  local hp_vectors="$4"
  local hp_ordering="$5"
  local spin_color_dilution="${6:-none}"

  export FLOWED_RINGED_DATA_DIR="$bench_root/data/$case_name"
  export FLOWED_RINGED_SM_TAG="S8T8_${case_name}"
  export FLOWED_RINGED_NOISE_SCHEME="$noise_scheme"
  export FLOWED_RINGED_N_VEC="$n_vec"
  export FLOWED_RINGED_HP_NUM_VECTORS="$hp_vectors"
  export FLOWED_RINGED_HP_ORDERING="$hp_ordering"
  export FLOWED_RINGED_SPIN_COLOR_DILUTION="$spin_color_dilution"
  export QUDA_RESOURCE_PATH="$bench_root/cache/$case_name"

  mkdir -p "$FLOWED_RINGED_DATA_DIR" "$QUDA_RESOURCE_PATH"
  local expected_h5="$FLOWED_RINGED_DATA_DIR/FlowedQuarkRinged/S8T8.FlowedQuarkRinged.${FLOWED_RINGED_CONFIG_NUM}.0.x0y0z0t0.S8T8_${case_name}.h5"

  echo "Running $case_name"
  echo "  gauge=$FLOWED_RINGED_GAUGE_PATH"
  echo "  ranks=$FLOWED_RINGED_NRANKS geometry=$FLOWED_RINGED_MPI_GEOMETRY"
  echo "  noise=$FLOWED_RINGED_NOISE_SCHEME n_vec=$FLOWED_RINGED_N_VEC hp=$FLOWED_RINGED_HP_NUM_VECTORS"
  echo "  hp_ordering=$FLOWED_RINGED_HP_ORDERING"
  echo "  spin_color_dilution=$FLOWED_RINGED_SPIN_COLOR_DILUTION"
  echo "  data=$FLOWED_RINGED_DATA_DIR"

  if [[ "${FLOWED_RINGED_SKIP_EXISTING:-0}" == "1" && -f "$expected_h5" ]]; then
    echo "  skipping existing $expected_h5"
  else
    /usr/bin/time -p /opt/cray/pals/1.8/bin/mpiexec -n "$FLOWED_RINGED_NRANKS" -envall \
      bash "$script_dir/run_flowed_quark_ringed_norm.sh" \
      > "$bench_root/log/${case_name}.o" \
      2> "$bench_root/log/${case_name}.e"
  fi
}

run_case "zn1024" "zn" "1024" "1" "interleaved_xyzt_binary_projected_to_evenodd" "none"
run_case "hp64x16" "hierarchical_probing" "64" "16" "interleaved_xyzt_binary_projected_to_evenodd" "none"
run_case "hp4x256" "hierarchical_probing" "4" "256" "interleaved_xyzt_binary_projected_to_evenodd" "none"
run_case "hp6x16sc12" "hierarchical_probing" "6" "16" "interleaved_xyzt_binary_projected_to_evenodd" "point"

python "$script_dir/analyze_s8t8_hp_convergence.py" --bench-root "$bench_root"

echo "Benchmark outputs:"
echo "  $bench_root/data"
echo "  $bench_root/log"
echo "  $bench_root/summary.csv"
echo "  $bench_root/summary.json"
