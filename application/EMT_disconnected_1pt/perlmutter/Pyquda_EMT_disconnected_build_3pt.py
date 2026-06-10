import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from pyquda_measurement_utils.io_corr import (
    get_emt_gluon_1pt_file_tag,
    get_emt_proton_2pt_file_tag,
    get_emt_quark_1pt_file_tag,
)


def parse_triplet(text):
    return [int(v) for v in text.split(".")]


def parse_int_list(text):
    return [int(v) for v in text.split(",") if v]


def parse_csv(text):
    return [v for v in text.split(",") if v]


def site_tag(src):
    return "x" + str(src[0]) + "y" + str(src[1]) + "z" + str(src[2]) + "t" + str(src[3])


def cfg_output_tag(configs):
    if len(configs) == 1:
        return str(configs[0])
    return f"{configs[0]}-{configs[-1]}-n{len(configs)}"


def default_loop_sm_tag():
    return os.environ.get("EMT_1PT_SM_TAG", "1HYP_GSRC_W1_k0")


def default_c2_sm_tag(interpolator):
    width = float(os.environ.get("EMT_DISC_WIDTH", "1.0"))
    return os.environ.get("EMT_DISC_SM_TAG", f"1HYP_GSRC_W{width:g}_k0_{interpolator}")


def infer_paths(kind, configs, data_dir, lat_tag, src_pos, loop_sm_tag, c2_sm_tag):
    paths = []
    for cfg in configs:
        if kind == "c2":
            tag = get_emt_proton_2pt_file_tag(data_dir, lat_tag, cfg, 0, src_pos, c2_sm_tag)
        elif kind == "quark":
            tag = get_emt_quark_1pt_file_tag(data_dir, lat_tag, cfg, 0, src_pos, loop_sm_tag)
        elif kind == "gluon":
            tag = get_emt_gluon_1pt_file_tag(data_dir, lat_tag, cfg, 0, src_pos, loop_sm_tag)
        else:
            raise ValueError(f"Unknown path kind {kind}")
        paths.append(tag + ".h5")
    return paths


def paths_from_env(env_name, fallback):
    text = os.environ.get(env_name, "")
    return parse_csv(text) if text else fallback


def require_files(paths, label):
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} file(s): {missing}")


def read_c2(path, gamma_label, momentum_label, t_separations):
    with h5py.File(path, "r") as h5:
        dataset = f"SS/{gamma_label}/{momentum_label}"
        if dataset not in h5:
            raise KeyError(f"{path} does not contain {dataset}")
        c2_t = h5[dataset][...]
    t_separations = np.asarray(t_separations, dtype=np.int32)
    if np.any(t_separations < 0) or np.any(t_separations >= c2_t.shape[0]):
        raise ValueError(f"Requested t_separations {t_separations.tolist()} outside C2 length {c2_t.shape[0]}")
    return c2_t[t_separations], c2_t


def read_quark_loop(path):
    with h5py.File(path, "r") as h5:
        raw = h5["raw/Tmunu_pervec"][...]
        volume_norm = h5.attrs.get("volume_norm")
        if volume_norm is None:
            raise KeyError(f"{path} is missing attrs/volume_norm")
        n_eff_attr = int(h5.attrs.get("effective_n_inversions", raw.shape[0]))
        if n_eff_attr != raw.shape[0]:
            raise ValueError(f"{path} has effective_n_inversions={n_eff_attr}, raw source axis={raw.shape[0]}")
        source_index = h5["raw/source_index"][...] if "raw/source_index" in h5 else np.arange(raw.shape[0], dtype=np.int32)
        base_noise_index = h5["raw/base_noise_index"][...] if "raw/base_noise_index" in h5 else np.zeros(raw.shape[0], dtype=np.int32)
        hp_index = h5["raw/hp_index"][...] if "raw/hp_index" in h5 else np.zeros(raw.shape[0], dtype=np.int32)
        qext = np.asarray(h5.attrs.get("qext", np.zeros((raw.shape[3], 4), dtype=np.int32)), dtype=np.int32)
        flow_times = np.asarray(h5.attrs.get("flow_times", np.arange(raw.shape[4])), dtype=np.float64)

    counts = np.arange(1, raw.shape[0] + 1, dtype=np.int32)
    cumulative = np.cumsum(raw, axis=0) / counts[:, None, None, None, None, None]
    cumulative = cumulative / float(volume_norm)
    bookkeeping = {
        "source_index": source_index,
        "base_noise_index": base_noise_index,
        "hp_index": hp_index,
    }
    return cumulative, counts, bookkeeping, qext, flow_times


def read_gluon_loop(path):
    with h5py.File(path, "r") as h5:
        sample = h5["Tmunu/T11"][...]
        loop = np.zeros((4, 4) + sample.shape, dtype=sample.dtype)
        for mu in range(4):
            for nu in range(mu, 4):
                data = h5[f"Tmunu/T{mu + 1}{nu + 1}"][...]
                loop[mu, nu] = data
                loop[nu, mu] = data
        qext = np.asarray(h5.attrs.get("qext", np.zeros((sample.shape[0], 4), dtype=np.int32)), dtype=np.int32)
        flow_times = np.asarray(h5.attrs.get("flow_times", np.arange(sample.shape[1])), dtype=np.float64)
    return loop, qext, flow_times


def zero_momentum_index(qext):
    if qext.ndim == 2:
        matches = np.where(np.all(qext[:, :3] == 0, axis=1))[0]
        if len(matches) > 0:
            return int(matches[0])
    return 0


def build_quark_products(c2, loops):
    c2_factor = c2[:, None, :, None, None, None, None, None]
    loop_factor = loops[:, :, None, :, :, :, :, :]
    c3_unsub = c2_factor * loop_factor
    c3_disc = None
    ratio = None
    if c2.shape[0] >= 2:
        mean_c2 = np.mean(c2, axis=0)
        mean_loop = np.mean(loops, axis=0)
        c3_disc = np.mean(c3_unsub, axis=0) - mean_c2[None, :, None, None, None, None, None] * mean_loop[:, None, :, :, :, :, :]
        ratio = c3_disc / mean_c2[None, :, None, None, None, None, None]
    return c3_unsub, c3_disc, ratio


def build_gluon_products(c2, loops):
    c2_factor = c2[:, :, None, None, None, None, None]
    loop_factor = loops[:, None, :, :, :, :, :]
    c3_unsub = c2_factor * loop_factor
    c3_disc = None
    ratio = None
    if c2.shape[0] >= 2:
        mean_c2 = np.mean(c2, axis=0)
        mean_loop = np.mean(loops, axis=0)
        c3_disc = np.mean(c3_unsub, axis=0) - mean_c2[:, None, None, None, None, None] * mean_loop[None, :, :, :, :, :]
        ratio = c3_disc / mean_c2[:, None, None, None, None, None]
    return c3_unsub, c3_disc, ratio


def write_string_dataset(group, name, values):
    group.create_dataset(name, data=np.asarray(values, dtype="S"))


parser = argparse.ArgumentParser()
parser.add_argument("--config_num", type=int, default=int(os.environ.get("EMT_1PT_CONFIG_NUM", "0")))
parser.add_argument("--interpolator", type=str, default=os.environ.get("EMT_DISC_INTERPOLATOR", "5"))
args, unknown = parser.parse_known_args()

configs = parse_int_list(os.environ.get("EMT_DISC_CONFIGS", str(args.config_num)))
data_dir = os.environ.get("EMT_1PT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
lat_tag = os.environ.get("EMT_1PT_LAT_TAG", "S8T32")
src_pos = parse_triplet(os.environ.get("EMT_1PT_SRC_POS", "0.0.0")) + [
    int(os.environ.get("EMT_1PT_SRC_T", "0"))
]
t_separations = parse_int_list(os.environ.get("EMT_DISC_T_SEPS", "2"))
c2_gamma = os.environ.get("EMT_DISC_C2_GAMMA", args.interpolator)
c2_momentum = os.environ.get("EMT_DISC_C2_MOMENTUM", "PX0PY0PZ0")
loop_sm_tag = default_loop_sm_tag()
c2_sm_tag = default_c2_sm_tag(args.interpolator)

c2_paths = paths_from_env("EMT_DISC_C2_FILES", infer_paths("c2", configs, data_dir, lat_tag, src_pos, loop_sm_tag, c2_sm_tag))
quark_paths = paths_from_env("EMT_DISC_QUARK_1PT_FILES", infer_paths("quark", configs, data_dir, lat_tag, src_pos, loop_sm_tag, c2_sm_tag))
gluon_paths = paths_from_env("EMT_DISC_GLUON_1PT_FILES", infer_paths("gluon", configs, data_dir, lat_tag, src_pos, loop_sm_tag, c2_sm_tag))

if not (len(c2_paths) == len(quark_paths) == len(gluon_paths) == len(configs)):
    raise ValueError("C2, quark 1pt, gluon 1pt, and config lists must have the same length")

require_files(c2_paths, "C2")
require_files(quark_paths, "quark 1pt")
require_files(gluon_paths, "gluon 1pt")

c2_selected = []
c2_full = []
quark_loops = []
gluon_loops = []
source_counts = []
source_indices = []
base_noise_indices = []
hp_indices = []
qext_list = []
quark_flow_time_list = []
gluon_qext_list = []
gluon_flow_time_list = []
for c2_path, quark_path, gluon_path in zip(c2_paths, quark_paths, gluon_paths):
    c2_tseps, c2_t = read_c2(c2_path, c2_gamma, c2_momentum, t_separations)
    quark_loop, source_count, source_bookkeeping, qext, quark_flow_times = read_quark_loop(quark_path)
    gluon_loop, gluon_qext, gluon_flow_times = read_gluon_loop(gluon_path)
    c2_selected.append(c2_tseps)
    c2_full.append(c2_t)
    quark_loops.append(quark_loop)
    gluon_loops.append(gluon_loop)
    source_counts.append(source_count)
    source_indices.append(source_bookkeeping["source_index"])
    base_noise_indices.append(source_bookkeeping["base_noise_index"])
    hp_indices.append(source_bookkeeping["hp_index"])
    qext_list.append(qext)
    quark_flow_time_list.append(quark_flow_times)
    gluon_qext_list.append(gluon_qext)
    gluon_flow_time_list.append(gluon_flow_times)

c2_selected = np.asarray(c2_selected)
c2_full = np.asarray(c2_full)
quark_loops = np.asarray(quark_loops)
gluon_loops = np.asarray(gluon_loops)
source_counts = np.asarray(source_counts)
source_indices = np.asarray(source_indices)
base_noise_indices = np.asarray(base_noise_indices)
hp_indices = np.asarray(hp_indices)

source_count = source_counts[0]
if not np.all(source_counts == source_count):
    raise ValueError("All quark 1pt files must use the same cumulative source counts")
if quark_loops.shape[1] != len(source_count):
    raise ValueError("Unexpected quark source-count axis mismatch")
for values, label in [
    (qext_list, "quark qext"),
    (gluon_qext_list, "gluon qext"),
    (quark_flow_time_list, "quark flow times"),
    (gluon_flow_time_list, "gluon flow times"),
]:
    reference = values[0]
    if any(not np.array_equal(reference, value) for value in values[1:]):
        raise ValueError(f"All input files must use matching {label}")
qext = qext_list[0]
quark_flow_times = quark_flow_time_list[0]
gluon_flow_times = gluon_flow_time_list[0]

quark_c3_unsub, quark_c3_disc, quark_ratio = build_quark_products(c2_selected, quark_loops)
gluon_c3_unsub, gluon_c3_disc, gluon_ratio = build_gluon_products(c2_selected, gluon_loops)

q0 = zero_momentum_index(qext)
t44_loop = quark_loops[:, :, 3, 3, q0, 0, :]
loop_norm = np.linalg.norm(t44_loop, axis=-1)
ratio_proxy = quark_c3_unsub[:, :, :, 3, 3, q0, 0, :] / c2_selected[:, None, :, None]
ratio_proxy = np.mean(ratio_proxy, axis=-1)

output = os.environ.get("EMT_DISC_3PT_OUT")
if output is None:
    cfg_tag = cfg_output_tag(configs)
    out_name = f"{lat_tag}.EMTdisc3pt.{cfg_tag}.0.{site_tag(src_pos)}.{loop_sm_tag}.c2_{c2_sm_tag}.h5"
    output = str(Path(data_dir) / "EMTdisc3pt" / out_name)

Path(output).parent.mkdir(parents=True, exist_ok=True)
with h5py.File(output, "w") as h5:
    h5.attrs["measurement"] = "proton_emt_disconnected_3pt_test"
    h5.attrs["ncfg"] = len(configs)
    h5.attrs["t_separations"] = np.asarray(t_separations, dtype=np.int32)
    h5.attrs["c2_gamma"] = c2_gamma
    h5.attrs["c2_momentum"] = c2_momentum
    h5.attrs["quark_has_stochastic_cumulative"] = True
    h5.attrs["vacuum_subtraction"] = "ensemble_only"
    h5.attrs["single_config_is_unsubtracted_proxy_only"] = len(configs) == 1
    h5.attrs["configs"] = np.asarray(configs, dtype=np.int32)
    h5.attrs["qext"] = qext
    h5.attrs["quark_flow_times"] = quark_flow_times
    h5.attrs["gluon_flow_times"] = gluon_flow_times
    h5.create_dataset("C2", data=c2_selected)
    h5.create_dataset("C2_full_time", data=c2_full)

    inputs = h5.require_group("inputs")
    write_string_dataset(inputs, "c2_files", c2_paths)
    write_string_dataset(inputs, "quark_1pt_files", quark_paths)
    write_string_dataset(inputs, "gluon_1pt_files", gluon_paths)

    qgrp = h5.require_group("quark")
    qgrp.create_dataset("source_count", data=source_count)
    qgrp.create_dataset("loop_cumulative", data=quark_loops)
    qgrp.create_dataset("C3_unsubtracted_cumulative", data=quark_c3_unsub)
    qgrp.create_dataset("source_index", data=source_indices)
    qgrp.create_dataset("base_noise_index", data=base_noise_indices)
    qgrp.create_dataset("hp_index", data=hp_indices)
    if quark_c3_disc is not None:
        qgrp.create_dataset("C3_disc_cumulative", data=quark_c3_disc)
        qgrp.create_dataset("R_disc_cumulative", data=quark_ratio)

    ggrp = h5.require_group("gluon")
    ggrp.create_dataset("loop", data=gluon_loops)
    ggrp.create_dataset("C3_unsubtracted", data=gluon_c3_unsub)
    if gluon_c3_disc is not None:
        ggrp.create_dataset("C3_disc", data=gluon_c3_disc)
        ggrp.create_dataset("R_disc", data=gluon_ratio)

    summary = h5.require_group("summary")
    summary.create_dataset("quark_source_count", data=source_count)
    summary.create_dataset("quark_T44_q0_flow0_loop_norm", data=loop_norm)
    summary.create_dataset("quark_T44_q0_flow0_unsub_ratio_proxy", data=ratio_proxy)
    summary.attrs["definition"] = json.dumps(
        {
            "quark_T44_q0_flow0_loop_norm": "L2 norm over tau of quark loop cumulative T44(q=0, flow=0)",
            "quark_T44_q0_flow0_unsub_ratio_proxy": "time average of C3_unsubtracted/C2 for T44(q=0, flow=0)",
        }
    )

print(f"Wrote disconnected 3pt diagnostic: {output}")
