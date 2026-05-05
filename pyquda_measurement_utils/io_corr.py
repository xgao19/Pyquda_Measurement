from pathlib import Path

import h5py
import numpy as np
import re

def get_sample_log_tag(ama, src, sm):

    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    log_sample = ama_tag + "_" + src_tag + "_" + sm_tag

    return log_sample

def get_c2pt_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".c2pt"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/c2pt/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag

def get_qTMD_file_tag(data_dir, lat, cfg, ama,src, sm):
    
    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMD"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qTMD/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag

def get_qTMDWF_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMDWF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qTMDWF/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag

def get_pion_EMFF_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".pion_EMFF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/pion_EMFF/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag

def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# EMT file-name helpers
# -----------------------------------------------------------------------------

def _emt_site_tag(src):
    return "x" + str(src[0]) + "y" + str(src[1]) + "z" + str(src[2]) + "t" + str(src[3])


def get_emt_gluon_1pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    return str(Path(data_dir) / "EMTg" / (str(lat) + ".EMTg." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm)))


def get_emt_quark_1pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    return str(Path(data_dir) / "EMTc" / (str(lat) + ".EMTc." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm)))


def get_emt_quark_3pt_file_tag(data_dir, lat, cfg, ama, src, sm, spin):
    return str(Path(data_dir) / "EMT3pt" / (str(lat) + ".EMT3pt." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm) + ".spin" + str(spin)))


def get_emt_meson_2pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    return str(Path(data_dir) / "EMT2pt" / (str(lat) + ".EMT2pt." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm)))


def get_emt_proton_2pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    return str(Path(data_dir) / "EMTproton2pt" / (str(lat) + ".EMTproton2pt." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm)))


def get_emt_proton_quark_3pt_file_tag(data_dir, lat, cfg, ama, src, sm, spin):
    return str(Path(data_dir) / "EMTproton3pt" / (str(lat) + ".EMTproton3pt." + str(cfg) + "." + str(ama) + "." + _emt_site_tag(src) + "." + str(sm) + ".spin" + str(spin)))
# -----------------------------------------------------------------------------
# EMT HDF5 writers
# -----------------------------------------------------------------------------


def _write_h5_attrs(obj, attrs):
    if not attrs:
        return
    for key, value in attrs.items():
        if value is None:
            continue
        obj.attrs[key] = value


def _prepare_h5_file(path, attrs=None):
    ensure_parent_dir(path)
    f = h5py.File(path, "w")
    _write_h5_attrs(f, attrs)
    return f


def save_emt_quark_1pt_hdf5(tag, Tmunu_pervec, CHI_pervec, Tmunu, CHI, attrs=None):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        raw = f.require_group("raw")
        raw.create_dataset("Tmunu_pervec", data=Tmunu_pervec)
        raw.create_dataset("CHI_pervec", data=CHI_pervec)

        avg = f.require_group("avg")
        avg.create_dataset("CHI", data=CHI)
        g_t = avg.require_group("Tmunu")
        for mu in range(4):
            for nu in range(mu, 4):
                g_t.create_dataset(f"T{mu+1}{nu+1}", data=Tmunu[mu, nu])


def save_emt_quark_3pt_hdf5(tag, C2, C3_chi, C3_Tmunu, momentum_transfer_list=None, attrs=None):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        f.create_dataset("C2", data=C2)
        f.create_dataset("C3_chi", data=C3_chi)
        f.create_dataset("C3_Tmunu", data=C3_Tmunu)
        if momentum_transfer_list is not None:
            f.create_dataset("momentum_transfer_list", data=np.asarray(momentum_transfer_list, dtype=np.int32))


def save_emt_meson_2pt_hdf5(tag, C2, gamma_list, momentum_list, attrs=None):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        f.create_dataset("C2", data=C2)
        f.create_dataset("gamma_list", data=np.asarray(gamma_list, dtype="S"))
        f.create_dataset("momentum_list", data=np.asarray(momentum_list, dtype=np.int32))


def save_emt_gluon_1pt_hdf5(tag, Tmunu_t, attrs=None):
    save_h5 = f"{tag}.h5"
    with _prepare_h5_file(save_h5, attrs) as f:
        g_t = f.require_group("Tmunu")
        for mu in range(4):
            for nu in range(mu, 4):
                g_t.create_dataset(f"T{mu+1}{nu+1}", data=Tmunu_t[mu, nu])


# -----------------------------------------------------------------------------
# Existing non-EMT HDF5 writers
# -----------------------------------------------------------------------------

def save_proton_c2pt_hdf5(corr, tag, gammalist, plist):

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
    sm = f.create_group("SS")
    for ig, gm in enumerate(gammalist):
        g = sm.create_group(gm)
        for ip, p in enumerate(plist):
            dataset_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            #print('DEBUG:', np.shape(corr), np.shape(gammalist), ig, ip)
            g.create_dataset(dataset_tag, data=np.roll(corr[ig][ip], roll, axis=0))
    f.close()

# W_index_list[bT, bz, eta, Tdir]
def save_qTMD_proton_hdf5_noRoll(corr, tag, gammalist, plist, W_index_list, tsep, latt_info):

    bT_list = ['b_X', 'b_Y']

    #g.message("-->>",W_index_list)

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')

    if latt_info.mpi_rank == 0:
        print(f"no roll")
        print(f"corr.shape, {np.shape(corr)}")
        print(f"plist.shape, {np.shape(plist)}")
    sm = f.require_group("SS")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.require_group(p_tag)
            for i, idx in enumerate(W_index_list):
                path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                g_data = g_p.require_group(path)
                g_data.create_dataset('bz'+str(idx[1]), data=corr[i][ip][ig][:tsep+2])
    f.close()

def save_qTMD_pion_hdf5_noRoll(corr, tag, gammalist, plist, W_index_list, tsep, latt_info):
    save_qTMD_proton_hdf5_noRoll(corr, tag, gammalist, plist, W_index_list, tsep, latt_info)

def save_pion_EMFF_hdf5_noRoll(corr, tag, gammalist, qlist, tsep, latt_info):

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')

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
