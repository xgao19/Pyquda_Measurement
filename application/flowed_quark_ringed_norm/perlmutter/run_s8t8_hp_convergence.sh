#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
software_root="${SOFTWARE_ROOT:-/global/cfs/cdirs/m3760/xgao/software}"
measurement_root="${MEASUREMENT_ROOT:-$software_root/Pyquda_Measurement}"

cd "$script_dir"
source "$measurement_root/systems/perlmutter/activate-venv-quda.sh"

export QUDA_PATH="${QUDA_PATH:-$software_root/quda/install}"
export QUDA_ENABLE_TUNING="${QUDA_ENABLE_TUNING:-0}"
export QUDA_ENABLE_MPS="${QUDA_ENABLE_MPS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

ranks="${FLOWED_RINGED_NRANKS:-4}"
mpi_geometry="${FLOWED_RINGED_MPI_GEOMETRY:-1.1.1.4}"
config_num="${FLOWED_RINGED_CONFIG_NUM:-0}"
gauge_path="${FLOWED_RINGED_GAUGE_PATH:-$measurement_root/test_gauge/S8T8_wilson_b6.0}"

bench_root="${FLOWED_RINGED_BENCH_ROOT:-$script_dir/benchmark/s8t8_hp_convergence}"
mkdir -p "$bench_root/log" "$bench_root/data" "$bench_root/cache" "$bench_root/cupy-cache"

run_case() {
  local case_name="$1"
  local noise_scheme="$2"
  local n_vec="$3"
  local hp_vectors="$4"
  local hp_ordering="$5"
  local spin_color_dilution="${6:-none}"

  local data_dir="$bench_root/data/$case_name"
  local quda_cache="$bench_root/cache/$case_name"
  local cupy_cache="$bench_root/cupy-cache/$case_name"
  local sm_tag="S8T8_${case_name}"

  mkdir -p "$data_dir" "$quda_cache" "$cupy_cache"
  local expected_h5="$data_dir/FlowedQuarkRinged/S8T8.FlowedQuarkRinged.${config_num}.0.x0y0z0t0.${sm_tag}.h5"

  echo "Running $case_name"
  echo "  gauge=$gauge_path"
  echo "  ranks=$ranks geometry=$mpi_geometry"
  echo "  noise=$noise_scheme n_vec=$n_vec hp=$hp_vectors"
  echo "  hp_ordering=$hp_ordering"
  echo "  spin_color_dilution=$spin_color_dilution"
  echo "  data=$data_dir"

  if [[ "${FLOWED_RINGED_SKIP_EXISTING:-0}" == "1" && -f "$expected_h5" ]]; then
    echo "  skipping existing $expected_h5"
  else
    /usr/bin/time -p srun --mpi=cray_shasta -n "$ranks" \
      --gpus-per-task="${FLOWED_RINGED_GPUS_PER_TASK:-1}" \
      env \
      FLOWED_RINGED_DATA_DIR="$data_dir" \
      FLOWED_RINGED_GAUGE_PATH="$gauge_path" \
      FLOWED_RINGED_CONFIG_NUM="$config_num" \
      FLOWED_RINGED_MPI_GEOMETRY="$mpi_geometry" \
      FLOWED_RINGED_SM_TAG="$sm_tag" \
      FLOWED_RINGED_NOISE_SCHEME="$noise_scheme" \
      FLOWED_RINGED_N_VEC="$n_vec" \
      FLOWED_RINGED_HP_NUM_VECTORS="$hp_vectors" \
      FLOWED_RINGED_HP_ORDERING="$hp_ordering" \
      FLOWED_RINGED_SPIN_COLOR_DILUTION="$spin_color_dilution" \
      QUDA_RESOURCE_PATH="$quda_cache" \
      CUPY_CACHE_DIR="$cupy_cache" \
      bash "$script_dir/run_flowed_quark_ringed_norm.sh" \
      > "$bench_root/log/${case_name}.o" \
      2> "$bench_root/log/${case_name}.e"
  fi
}

run_case "zn1024" "zn" "1024" "1" "interleaved_xyzt_binary_projected_to_evenodd" "none"
run_case "hp64x16" "hierarchical_probing" "64" "16" "interleaved_xyzt_binary_projected_to_evenodd" "none"
run_case "hp4x256" "hierarchical_probing" "4" "256" "interleaved_xyzt_binary_projected_to_evenodd" "none"
run_case "hp6x16sc12" "hierarchical_probing" "6" "16" "interleaved_xyzt_binary_projected_to_evenodd" "point"

python3 "$script_dir/analyze_s8t8_hp_convergence.py" --bench-root "$bench_root"
python3 "$script_dir/plot_s8t8_hp_convergence_pdf.py" --bench-root "$bench_root"

echo "Benchmark outputs:"
echo "  $bench_root/data"
echo "  $bench_root/log"
echo "  $bench_root/summary.csv"
echo "  $bench_root/summary.json"
echo "  $bench_root/s8t8_hp_convergence_results.pdf"
