from pathlib import Path

import h5py
import numpy as np

from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
    basis_attrs,
    basis_metadata,
)
import re


# -----------------------------------------------------------------------------
# Shared tag/path helpers
# -----------------------------------------------------------------------------

# Build a compact sample identifier for log files.
def get_sample_log_tag(ama, src, sm):

    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    log_sample = ama_tag + "_" + src_tag + "_" + sm_tag

    return log_sample


def get_spatial_vector_tag(vector):
    """Return a compact x/y/z tag for a three-component integer vector."""
    if len(vector) != 3:
        raise ValueError("spatial vector tags require exactly three components")
    return f"x{int(vector[0])}y{int(vector[1])}z{int(vector[2])}"


def get_interpolator_channel_tag(sm, src_interpolator, sink_interpolator=None):
    """Append explicit source/sink interpolator identity to a setup tag."""
    tag = f"{sm}.src{src_interpolator}"
    if sink_interpolator is not None:
        tag += f".sink{sink_interpolator}"
    return tag


def get_pion_channel_tag(sm, src_interpolator, sink_interpolator=None):
    """Append explicit pion source/sink interpolator identity to a setup tag."""
    return get_interpolator_channel_tag(
        sm, src_interpolator, sink_interpolator
    )


# Build the standard point-source two-point output tag.
def get_c2pt_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".c2pt"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/c2pt/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag


# Build the standard qTMD output tag used by proton and pion qTMD applications.
def get_qTMD_file_tag(data_dir, lat, cfg, ama,src, sm):
    
    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMD"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qTMD/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag


# Build the source-independent disconnected qTMD loop output tag.
def get_disconnected_qTMD_loop_file_tag(data_dir, lat, cfg, ama, sm):
    return str(Path(data_dir) / "qTMD1pt" / (str(lat) + ".qTMD1pt." + str(cfg) + "." + str(ama) + "." + str(sm)))


# Build the standard qTMDWF output tag.
def get_qTMDWF_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMDWF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qTMDWF/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag


# -----------------------------------------------------------------------------
# Pion EMFF tag helpers
# -----------------------------------------------------------------------------

# Build the pion electromagnetic form-factor output tag.
def get_pion_EMFF_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".pion_EMFF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/pion_EMFF/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag


# -----------------------------------------------------------------------------
# Pion soft-factor tag helpers
# -----------------------------------------------------------------------------

def _pion_soft_factor_tag(data_dir, kind, lat, cfg, ama, source_time, sm, quark_momenta_z):
    qz = np.asarray(quark_momenta_z)
    if qz.ndim != 1 or not len(qz) or qz.dtype.kind not in "iu" or len(set(qz.tolist())) != len(qz):
        raise ValueError("quark_momenta_z must be a nonempty list of distinct integers")
    if not isinstance(source_time, (int, np.integer)) or source_time < 0:
        raise ValueError("source_time must be a nonnegative integer")
    qtag = "qz" + "_".join(str(int(q)) for q in sorted(qz))
    name = f"{lat}.{kind}.{cfg}.{ama}.wallt{source_time}.{sm}.{qtag}"
    return str(Path(data_dir) / kind / str(cfg) / name)


def get_pion_soft_factor_file_tag(data_dir, lat, cfg, ama, source_time, sm, quark_momenta_z, t_sep):
    """Name one dense four-point file containing all ordered quark pairs."""
    if not isinstance(t_sep, (int, np.integer)) or t_sep <= 0:
        raise ValueError("t_sep must be a positive integer")
    return _pion_soft_factor_tag(data_dir, "pion_soft_factor", lat, cfg, ama,
                                source_time, sm, quark_momenta_z) + f".dt{t_sep}"


def get_pion_soft_factor_qTMDWF_file_tag(data_dir, lat, cfg, ama, source_time, sm, quark_momenta_z):
    """Name the wall-source SP collection for one source time."""
    return _pion_soft_factor_tag(data_dir, "pion_soft_factor_qTMDWF", lat, cfg, ama,
                                source_time, sm, quark_momenta_z)


def get_pion_soft_factor_c2pt_file_tag(data_dir, lat, cfg, ama, source_time, sm, quark_momenta_z):
    """Name the wall-source SS collection for one source time."""
    return _pion_soft_factor_tag(data_dir, "pion_soft_factor_c2pt", lat, cfg, ama,
                                source_time, sm, quark_momenta_z)


# Build the saved wall-source propagator tag for the pion soft-factor workflow.
def get_pion_soft_factor_prop_file_tag(data_dir, lat, cfg, ama, src, sm, quarkmom):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".pion_soft_factor_prop"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    mom_tag = "qx"+str(quarkmom[0]) + "qy"+str(quarkmom[1]) + "qz"+str(quarkmom[2])
    sm_tag  = str(sm)

    return data_dir + "/pion_soft_factor_prop/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag + "." + mom_tag


# Ensure the parent directory exists before opening an HDF5 file.
def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# EMT file-name helpers
# -----------------------------------------------------------------------------

# Build the canonical EMT source-position tag.
def _emt_site_tag(src):
    return "x" + str(src[0]) + "y" + str(src[1]) + "z" + str(src[2]) + "t" + str(src[3])


# Build the canonical source-independent gluon EMT loop output tag.
def get_emt_gluon_loop_file_tag(data_dir, lat, cfg, ama, sm):
    return str(Path(data_dir) / "EMTgluonLoop" / (str(lat) + ".EMTgluonLoop." + str(cfg) + "." + str(ama) + "." + str(sm)))


# Build the canonical source-independent quark EMT loop output tag.
def get_emt_quark_loop_file_tag(data_dir, lat, cfg, ama, sm):
    return str(Path(data_dir) / "EMTquarkLoop" / (str(lat) + ".EMTquarkLoop." + str(cfg) + "." + str(ama) + "." + str(sm)))


# Build the pion/meson quark EMT three-point output tag.
def get_emt_quark_3pt_file_tag(
    data_dir, lat, cfg, ama, src, sm, src_interpolator, sink_interpolator
):
    name = (
        f"{lat}.EMT3pt.{cfg}.{ama}.{_emt_site_tag(src)}.{sm}"
        f".src{src_interpolator}.sink{sink_interpolator}"
    )
    return str(Path(data_dir) / "EMT3pt" / name)


# Build the pion/meson EMT two-point diagnostic output tag.
def get_emt_meson_2pt_file_tag(
    data_dir, lat, cfg, ama, src, sm, src_interpolator
):
    name = (
        f"{lat}.EMT2pt.{cfg}.{ama}.{_emt_site_tag(src)}.{sm}"
        f".src{src_interpolator}"
    )
    return str(Path(data_dir) / "EMT2pt" / name)


# Build the proton EMT two-point diagnostic output tag.
def get_emt_proton_2pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    return str(Path(data_dir) / "EMTproton2pt" / (str(lat) + ".EMTproton2pt." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm)))


# Build the proton quark EMT three-point output tag.
def get_emt_proton_quark_3pt_file_tag(data_dir, lat, cfg, ama, src, sm, pf, t_sep):
    if isinstance(t_sep, (list, tuple, np.ndarray)):
        if len(t_sep) != 1:
            raise ValueError("proton EMT 3pt tags encode exactly one t_sep")
        t_sep = t_sep[0]
    tsep_tag = str(int(t_sep))
    kinematics = f"PX{pf[0]}PY{pf[1]}PZ{pf[2]}dt{tsep_tag}"
    return str(Path(data_dir) / "EMTproton3pt" / (str(lat) + ".EMTproton3pt." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm) + "." + kinematics))


# Build the flowed-quark ringed-normalization output tag.
def get_flowed_quark_ringed_norm_file_tag(data_dir, lat, cfg, ama, sm):
    return str(Path(data_dir) / "FlowedQuarkRinged" / (str(lat) + ".FlowedQuarkRinged." + str(cfg) + "." + str(ama) + "." + str(sm)))


# -----------------------------------------------------------------------------
# Shared HDF5 helpers
# -----------------------------------------------------------------------------


# Attach optional metadata to an HDF5 file or group.
def _write_h5_attrs(obj, attrs):
    if not attrs:
        return
    for key, value in attrs.items():
        if value is None:
            continue
        obj.attrs[key] = value


# Open a fresh HDF5 file after creating its parent directory.
def _prepare_h5_file(path, attrs=None):
    ensure_parent_dir(path)
    f = h5py.File(path, "w")
    _write_h5_attrs(f, attrs)
    return f


# -----------------------------------------------------------------------------
# EMT HDF5 writers
# -----------------------------------------------------------------------------

# Save quark EMT three-point functions without embedding two-point data.
def save_emt_quark_3pt_hdf5(
    tag,
    C3_chi,
    C3_Tmunu,
    C3_local_bilinear,
    C3_derivative_bilinear,
    momentum_transfer_list=None,
    attrs=None,
):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        f.create_dataset("C3_chi", data=C3_chi)
        f.create_dataset("C3_Tmunu", data=C3_Tmunu)
        f.create_dataset("C3_local_bilinear", data=C3_local_bilinear)
        f.create_dataset("C3_derivative_bilinear", data=C3_derivative_bilinear)
        for name, values in basis_metadata().items():
            f.create_dataset(name, data=values)
        if momentum_transfer_list is not None:
            f.create_dataset("momentum_transfer_list", data=np.asarray(momentum_transfer_list, dtype=np.int32))


# Save pion/meson EMT two-point functions and their gamma/momentum metadata.
def save_emt_meson_2pt_hdf5(tag, C2, gamma_list, momentum_list, attrs=None):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        f.create_dataset("C2", data=C2)
        f.create_dataset("gamma_list", data=np.asarray(gamma_list, dtype="S"))
        f.create_dataset("momentum_list", data=np.asarray(momentum_list, dtype=np.int32))


def get_topological_charge_loop_file_tag(data_dir, ensemble, stream, cfg, setup):
    """Build a source-independent tag including stream and configuration."""
    return str(Path(data_dir) / "topologicalChargeLoop" / f"{stream}.{cfg}" /
               f"{ensemble}.topologicalChargeLoop.{stream}.{cfg}.{setup}")


def save_topological_charge_1pt_hdf5(tag, result):
    """Write scalar topology loops; the caller must be MPI rank zero."""
    with _prepare_h5_file(f"{tag}.h5", result["attrs"]) as handle:
        handle.create_dataset("loop", data=np.asarray(result["loop"], dtype=np.complex128))
        handle.create_dataset("momentum_list", data=np.asarray(result["momentum_list"], dtype=np.int32))
        handle.create_dataset("flow_times_lattice", data=result["flow_times_lattice"])
        handle.create_dataset("diagnostics/Q_total", data=result["Q_total"])


# Save flowed gluon EMT one-point data.
def save_emt_gluon_1pt_hdf5(tag, Tmunu_t, attrs=None):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        g_t = f.require_group("Tmunu")
        g_t.attrs["upper_triangle_only"] = True
        for mu in range(4):
            for nu in range(mu, 4):
                g_t.create_dataset(f"T{mu+1}{nu+1}", data=Tmunu_t[mu, nu])


# -----------------------------------------------------------------------------
# Two-point and qTMD HDF5 writers
# -----------------------------------------------------------------------------

# Save the standard baryon-style two-point function with source-time rolling.
def save_proton_c2pt_hdf5(
    corr,
    tag,
    gammalist,
    plist,
    attrs=None,
    write_gamma_basis=False,
):

    src_match = None
    for part in tag.split("."):
        src_match = re.search(r"^x-?\d+y-?\d+z-?\d+t(-?\d+)$", part)
        if src_match is not None:
            break
    if src_match is None:
        raise ValueError(f"Could not parse source time from c2pt tag: {tag}")
    roll = -int(src_match.group(1))

    save_h5 = tag + ".h5"
    ensure_parent_dir(save_h5)
    f = h5py.File(save_h5, 'w')
    _write_h5_attrs(f, attrs)
    if write_gamma_basis:
        for name, values in basis_metadata().items():
            f.create_dataset(name, data=values)
    sm = f.create_group("SS")
    for ig, gm in enumerate(gammalist):
        g = sm.create_group(gm)
        for ip, p in enumerate(plist):
            dataset_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            #print('DEBUG:', np.shape(corr), np.shape(gammalist), ig, ip)
            g.create_dataset(dataset_tag, data=np.roll(corr[ig][ip], roll, axis=0))
    f.close()


# Save one connected qTMD/PDF operator with dense Wilson, momentum, Gamma,
# and source-relative time axes.  Applications call this writer only on rank 0.
def save_connected_qtmd_hdf5(
    corr,
    tag,
    momentum_list,
    wilson_index_list,
    t_sep,
    attrs=None,
):
    corr = np.asarray(corr)
    momentum_list = np.asarray(momentum_list, dtype=np.int32)
    wilson_index_list = np.asarray(wilson_index_list, dtype=np.int32)
    t_sep = int(t_sep)
    expected_shape = (
        len(wilson_index_list),
        len(momentum_list),
        len(GAMMA_LABELS),
        t_sep + 2,
    )
    if corr.shape != expected_shape:
        raise ValueError(
            "connected qTMD correlation should have axes "
            f"[wilson,momentum,gamma,time]={expected_shape}, got {corr.shape}"
        )
    if momentum_list.ndim != 2 or momentum_list.shape[1] != 4:
        raise ValueError("momentum_list should have shape [N_momentum,4]")
    if wilson_index_list.ndim != 2 or wilson_index_list.shape[1] != 4:
        raise ValueError("wilson_index_list should have shape [N_wilson,4]")

    file_attrs = {
        **basis_attrs(),
        "qtmd_hdf5_schema": "connected_qtmd_dense_v1",
        "corr_axes": "wilson,momentum,gamma,time",
        "momentum_columns": "px,py,pz,energy",
        "wilson_index_columns": "bT,bz,eta,transverse_direction",
        "time_axis": "source_relative",
        "t_separation": t_sep,
    }
    if attrs:
        file_attrs.update(attrs)

    with _prepare_h5_file(f"{tag}.h5", file_attrs) as f:
        f.create_dataset("corr", data=corr)
        f.create_dataset("momentum_list", data=momentum_list)
        f.create_dataset("wilson_index_list", data=wilson_index_list)
        for name, values in basis_metadata().items():
            f.create_dataset(name, data=values)


# -----------------------------------------------------------------------------
# Pion EMFF HDF5 writers
# -----------------------------------------------------------------------------

# Save pion electromagnetic form-factor three-point data.
def save_pion_EMFF_hdf5_noRoll(
    corr, tag, gammalist, qlist, tsep, latt_info, attrs=None
):

    save_h5 = tag + ".h5"
    ensure_parent_dir(save_h5)
    f = h5py.File(save_h5, 'w')
    _write_h5_attrs(f, attrs)

    if latt_info.mpi_rank == 0:
        print(f"no roll")
        print(f"corr.shape, {np.shape(corr)}")
        print(f"qlist.shape, {np.shape(qlist)}")
    sm = f.require_group("SS")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for iq, q in enumerate(qlist):
            q_tag = "PX"+str(q[0])+"PY"+str(q[1])+"PZ"+str(q[2])
            g_gm.create_dataset(q_tag, data=corr[iq][ig][:tsep+2])
    f.close()


# -----------------------------------------------------------------------------
# Pion soft-factor HDF5 writers
# -----------------------------------------------------------------------------

# Save the pion soft-factor four-point correlator.
def _matrix_for_hdf5(matrix):
    matrix = matrix.matrix if hasattr(matrix, "matrix") else matrix
    if hasattr(matrix, "get"):
        matrix = matrix.get()
    elif type(matrix).__module__.split(".")[0] == "dpnp":
        import dpnp

        matrix = dpnp.asnumpy(matrix)
    return np.asarray(matrix)


PION_SOFT_FACTOR_STORAGE_SCHEMA = "pion_soft_factor_dense_v1"


def _soft_factor_dense_inputs(corr, quark_momentum_pairs, channel_count, displacements=None):
    pairs = np.asarray(quark_momentum_pairs)
    if pairs.ndim != 3 or pairs.shape[1:] != (2, 3) or not len(pairs) or pairs.dtype.kind not in "iu":
        raise ValueError("quark_momentum_pairs must have integer shape [pair,fw_bw,xyz]")
    if len(np.unique(pairs.reshape(len(pairs), -1), axis=0)) != len(pairs):
        raise ValueError("Duplicate ordered quark momentum pairs")
    data = np.asarray(corr, dtype=np.complex128)
    shape = (len(pairs), channel_count)
    vectors = None
    if displacements is not None:
        vectors = np.asarray(displacements)
        if vectors.ndim != 2 or vectors.shape[1] != 3 or not len(vectors) or vectors.dtype.kind not in "iu":
            raise ValueError("displacements must have integer shape [displacement,xyz]")
        shape += (len(vectors),)
    if data.ndim != len(shape) + 1 or data.shape[:-1] != shape or data.shape[-1] == 0:
        raise ValueError(f"Dense correlator shape must be {shape} + (time,), got {data.shape}")
    if not channel_count or not np.all(np.isfinite(data)):
        raise ValueError("Dense correlator channels must be nonempty and data finite")
    return data, pairs.astype(np.int32), None if vectors is None else vectors.astype(np.int32)


def _pion_matrix_metadata(pion_channel_pairs):
    labels = list(pion_channel_pairs)
    source = np.stack([_matrix_for_hdf5(pion_channel_pairs[label][0]) for label in labels])
    sink = np.stack([_matrix_for_hdf5(pion_channel_pairs[label][1]) for label in labels])
    if (source.shape != (len(labels), 4, 4) or sink.shape != source.shape
            or not np.all(np.isfinite(source)) or not np.all(np.isfinite(sink))):
        raise ValueError("Pion source and sink matrices must have shape [channel,4,4]")
    return {"pion_pair_labels": np.asarray(labels, dtype="S"),
            "pion_source_matrices": source, "pion_sink_matrices": sink}


def _write_pion_soft_factor_metadata(handle, definition, axes, pairs, nt, attrs):
    _write_h5_attrs(handle, attrs)
    _write_h5_attrs(handle, {
        **basis_attrs(),
        "soft_factor_schema": "full_spin_color_v3",
        "soft_factor_storage_schema": PION_SOFT_FACTOR_STORAGE_SCHEMA,
        "correlator_definition": definition,
        "dataset_axes": axes,
        "gamma_convention": "canonical_raw_pyquda",
        "shift_convention": "positive_displacement_reads_x_plus_b",
        "time_axis": "source_relative", "time_origin": "relative_to_source",
        "source_type": "wall", "volume_normalization": "none",
        "quark_momentum_pair_axes": "pair,fw_bw,xyz",
        "quark_line_order": "fw,bw", "displacement_columns": "dx,dy,dz",
        "momentum_units": "2*pi/L", "backward_propagator_momentum": "negative_bw",
    })
    if "lattice" in handle.attrs and int(handle.attrs["lattice"][3]) != nt:
        raise ValueError("Time extent differs from lattice metadata")
    for name, values in basis_metadata().items():
        handle.create_dataset(name, data=values)
    handle.create_dataset("quark_momentum_pairs", data=pairs)
    handle.create_dataset("meson_momentum_list", data=pairs.sum(axis=1))
    handle.create_dataset("time", data=np.arange(nt, dtype=np.int32))


def save_pion_soft_factor_hdf5_noRoll(corr, tag, quark_momentum_pairs,
                                      pion_channel_pairs, gamma_channel_pairs,
                                      displacements, t_sep, attrs=None):
    """Write C4[pair,gamma_pair,displacement,time] for exactly one tsep."""
    if len(pion_channel_pairs) != 1:
        raise ValueError("A dense C4 file requires one fixed pion source/sink channel")
    if not isinstance(t_sep, (int, np.integer)) or t_sep <= 0:
        raise ValueError("t_sep must be a positive integer")
    data, pairs, vectors = _soft_factor_dense_inputs(
        corr, quark_momentum_pairs, len(gamma_channel_pairs), displacements)
    metadata = _pion_matrix_metadata(pion_channel_pairs)
    gamma_labels = list(gamma_channel_pairs)
    gamma1 = [gamma_channel_pairs[label][0] for label in gamma_labels]
    gamma2 = [gamma_channel_pairs[label][1] for label in gamma_labels]
    if any(label not in GAMMA_LABELS for label in gamma1 + gamma2):
        raise ValueError("Unknown canonical raw Gamma label")
    with _prepare_h5_file(str(tag) + ".h5") as handle:
        _write_pion_soft_factor_metadata(handle, "sum_x_trace_A_Gamma2_B_Gamma1",
            "quark_pair,gamma_pair,displacement,time", pairs, data.shape[-1], attrs)
        handle.attrs["t_sep"] = int(t_sep)
        handle.create_dataset("C4", data=data)
        handle.create_dataset("displacement_vectors", data=vectors)
        for name, value in metadata.items():
            handle.create_dataset(name, data=value)
        handle.create_dataset("gamma_pair_labels", data=np.asarray(gamma_labels, dtype="S"))
        handle.create_dataset("gamma1_labels", data=np.asarray(gamma1, dtype="S"))
        handle.create_dataset("gamma2_labels", data=np.asarray(gamma2, dtype="S"))


def save_pion_soft_factor_qTMDWF_hdf5_noRoll(corr, tag, quark_momentum_pairs,
                                            source_label, source_matrix,
                                            displacements, attrs=None):
    """Write SP[pair,sink_gamma,displacement,time] without an extra time roll."""
    data, pairs, vectors = _soft_factor_dense_inputs(
        corr, quark_momentum_pairs, len(GAMMA_LABELS), displacements)
    matrix = _matrix_for_hdf5(source_matrix)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Pion source matrix must have shape [4,4]")
    with _prepare_h5_file(str(tag) + ".h5") as handle:
        _write_pion_soft_factor_metadata(handle, "SP_sum_x_trace_source_bar_backward_sink_forward",
            "quark_pair,sink_gamma,displacement,time", pairs, data.shape[-1], attrs)
        handle.attrs["source_interpolator"] = str(source_label)
        handle.create_dataset("SP", data=data)
        handle.create_dataset("pion_source_matrix", data=matrix)
        handle.create_dataset("sink_gamma_labels", data=np.asarray(GAMMA_LABELS, dtype="S"))
        handle.create_dataset("displacement_vectors", data=vectors)


def save_pion_soft_factor_c2pt_hdf5_noRoll(corr, tag, quark_momentum_pairs,
                                         pion_channel_pairs, attrs=None):
    """Write SS[pair,pion_pair,time] for all wall sink channels."""
    data, pairs, _ = _soft_factor_dense_inputs(corr, quark_momentum_pairs, len(pion_channel_pairs))
    metadata = _pion_matrix_metadata(pion_channel_pairs)
    with _prepare_h5_file(str(tag) + ".h5") as handle:
        _write_pion_soft_factor_metadata(handle, "SS_trace_product_of_separately_projected_spatial_sums",
            "quark_pair,pion_pair,time", pairs, data.shape[-1], attrs)
        handle.create_dataset("SS", data=data)
        for name, value in metadata.items():
            handle.create_dataset(name, data=value)


# -----------------------------------------------------------------------------
# qTMDWF HDF5 writers
# -----------------------------------------------------------------------------

# Save the standard qTMDWF output after the application has already rolled time.
def save_qTMDWF_hdf5_noRoll(corr, tag, gammalist, plist, W_index_list):

    bT_list = ['b_X', 'b_Y']

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')

    sm = f.require_group("SP")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.require_group(p_tag)
            for i, idx in enumerate(W_index_list):
                path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                g_data = g_p.require_group(path)
                g_data.create_dataset('bz'+str(idx[1]), data=corr[i][ip][ig])
    f.close()
