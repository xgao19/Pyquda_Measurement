"""Kinetic-only flowed-quark measurement built on the EMT shared runner."""

import os

import h5py
import numpy as np
from opt_einsum import contract

from pyquda_measurement_utils.Disconnected_1pt_EMT_vibe_develop import (
    EMTDisconnectedQuark1pt,
    _flow_times,
)
from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA5_HERMITICITY_PARTNERS,
    GAMMA5_HERMITICITY_SIGNS,
    VECTOR_GAMMA_POSITIONS,
)
from pyquda_measurement_utils.Disconnected_utils_vibe_develop import (
    COUNTER_NOISE_ALGORITHM,
    canonical_temp_path,
    discover_shard_layout,
    iter_validated_shard_parts,
)


class RingedQuark1pt(EMTDisconnectedQuark1pt):
    """Measure only the ringed kinetic trace using EMT production machinery."""

    def __init__(self, parameters):
        parameters = dict(parameters)
        if parameters.get("config_num") is None:
            raise ValueError(
                "config_num is required for counter-based stochastic sources"
            )
        qext = parameters.get("qext", [[0, 0, 0, 0]])
        if not np.array_equal(np.asarray(qext), np.asarray([[0, 0, 0, 0]])):
            raise ValueError("RingedQuark1pt supports only the unique zero momentum")
        parameters["qext"] = [[0, 0, 0, 0]]
        super().__init__(parameters)

    def _raw_step_tail_shapes(self, latt_info):
        return {"kinetic_pervec": (int(latt_info.global_size[3]),)}

    def _metadata_datasets(self):
        return {}

    def _output_kind(self):
        return "flowed_quark_ringed_norm"

    def _completion_label(self):
        return "ringed"

    def _contract_flowed_source(
        self, U_f, gauge_dirac, xi, eta, momentum_projectors
    ):
        """Compute the exact time-slice kinetic trace from two-sided EMT loops."""
        vector_gammas = self._vector_gamma_stack_for(eta.data)
        right_field_sum = None
        for mu in range(4):
            derivative_right = gauge_dirac.covDev(eta, mu)
            derivative_left = gauge_dirac.covDev(eta, mu + 4)
            derivative = derivative_right - derivative_left
            term = contract(
                "wtzyxia,ij,wtzyxja->wtzyx",
                xi.data.conj(), vector_gammas[mu], derivative.data,
            )
            gamma_position = VECTOR_GAMMA_POSITIONS[mu]
            partner = GAMMA5_HERMITICITY_PARTNERS[gamma_position]
            if partner != gamma_position:
                raise RuntimeError(
                    "vector Gamma should map to itself under gamma5 hermiticity"
                )
            if GAMMA5_HERMITICITY_SIGNS[gamma_position] != -1:
                raise RuntimeError("vector Gamma should be odd under gamma5 hermiticity")
            right_field_sum = (
                term if right_field_sum is None else right_field_sum + term
            )
            del (
                term,
                derivative,
                derivative_right,
                derivative_left,
            )

        right_projected = self._project_gamma_fields(
            right_field_sum[None, ...], momentum_projectors.derivative_phases
        )
        closed_loop = self._closed_loop_derivative_from_right_projection(
            right_projected,
            momentum_projectors,
            gamma_partners=(0,),
            gamma_signs=(-1,),
        )[0, 0]
        projected = np.asarray(-2.0 * closed_loop)
        if projected.shape != (int(U_f.latt_info.global_size[3]),):
            raise RuntimeError(
                f"unexpected ringed kinetic time shape {projected.shape}"
            )
        spatial_volume = int(np.prod(U_f.latt_info.global_size[:3]))
        return {"kinetic_pervec": projected / spatial_volume}

    def _measurement_attrs(
        self,
        latt_info,
        invPara,
        randPara,
        counter_config,
        counter_stream,
        n_eff,
        spatial_volume,
    ):
        n_vec, n_zn, _ = randPara
        mass, csw, _, _ = invPara
        return {
            "measurement": "flowed_quark_ringed_norm",
            "content": "kinetic_only",
            "producer": "standalone_ringed_shared_emt_runner",
            "operator": "bar_chi_overleftrightarrow_Dslash_chi",
            "kinetic_pervec_axes": "source,flow,t",
            "kinetic_relation_to_emt": (
                "K=-2*sum_mu(L_D[gamma_mu,mu,q0])/spatial_volume"
            ),
            "derivative_convention": (
                "two_sided_overleftrightarrow_D_from_gamma5_hermiticity"
            ),
            "gamma5_hermiticity_reconstruction": True,
            "derivative_closed_fermion_loop_sign_included": True,
            "flow_type": self.flow_type,
            "flow_epsilon": self.flow_epsilon,
            "flow_steps": self.flow_steps,
            "flow_times": _flow_times(self.flow_epsilon, self.flow_steps),
            "qext": np.asarray(self.qlist, dtype=np.int32),
            "volume_norm": int(spatial_volume),
            "mass": mass,
            "csw": csw,
            "gauge_preprocessing": self.gauge_preprocessing,
            "t_boundary": latt_info.t_boundary,
            "flavor_convention": self.flavor_convention,
            "n_vec": n_vec,
            "n_base_noise": n_vec,
            "effective_n_inversions": n_eff,
            "n_zn": n_zn,
            "config_num": int(counter_config),
            "noise_stream": int(counter_stream),
            "noise_generator": COUNTER_NOISE_ALGORITHM,
            "noise_counter_order": "global_xyzt_spin_color_config_base_stream",
            "noise_scheme": self.noise_scheme,
            "hp_num_vectors": self.hp_num_vectors,
            "hp_ordering": self.hp_ordering,
            "ringed_factors_stored": False,
        }

    def measure(
        self,
        gauge,
        invPara,
        randPara,
        tag,
        *,
        shard_dir=None,
        sample_log_file=None,
        base_start=0,
        base_stop=None,
        block_interval_solves=64,
        flow_batch_size=1,
    ):
        """Run the kinetic-only standalone measurement into base/HP shards."""
        return self._run_sharded_measurement(
            gauge,
            invPara,
            randPara,
            tag=tag,
            shard_dir=shard_dir,
            sample_log_file=sample_log_file,
            base_start=base_start,
            base_stop=base_stop,
            block_interval_solves=block_interval_solves,
            flow_batch_size=flow_batch_size,
        )


def finalize_ringed_quark_1pt_shards(shard_dir, canonical_tag, n_base_noise):
    """Stream complete kinetic-only shards into one canonical ringed file."""
    manifest = discover_shard_layout(
        shard_dir,
        canonical_tag,
        n_base_noise,
        raw_dataset_names=("kinetic_pervec",),
    )
    attrs = {
        key: value
        for key, value in manifest["reference_attrs"].items()
        if key not in {
            "shard_schema",
            "output_kind",
            "block_interval_solves",
            "hp_vectors_per_base",
        }
    }
    total_sources = int(manifest["total_sources"])
    attrs.update({
        "measurement": "flowed_quark_ringed_norm",
        "content": "kinetic_only",
        "n_vec": int(n_base_noise),
        "n_base_noise": int(n_base_noise),
        "effective_n_inversions": total_sources,
        "ringed_factors_stored": False,
    })
    kinetic_tail = manifest["raw_tails"]["kinetic_pervec"]
    if len(kinetic_tail) != 2:
        raise ValueError(
            "ringed kinetic shards should have tail shape [Nflow,Nt], "
            f"got {kinetic_tail}"
        )
    nt = kinetic_tail[-1]
    final_path, temp_path = canonical_temp_path(canonical_tag)
    with h5py.File(temp_path, "w") as out:
        for key, value in attrs.items():
            out.attrs[key] = value
        out.create_dataset(
            "flow_times", data=np.asarray(attrs["flow_times"], dtype=np.float64)
        )
        raw = out.require_group("raw")
        kinetic_out = raw.create_dataset(
            "kinetic_pervec",
            shape=(total_sources,) + kinetic_tail,
            dtype=np.complex128,
        )
        bookkeeping = {
            name: raw.create_dataset(name, shape=(total_sources,), dtype=np.int32)
            for name in ("base_noise_index", "hp_index")
        }
        kinetic_sum = np.zeros(kinetic_tail[:-1], dtype=np.complex128)
        for info, part in iter_validated_shard_parts(manifest):
            start, stop = info["output_start"], info["output_stop"]
            values = part["raw/kinetic_pervec"][()]
            kinetic_out[start:stop] = values
            kinetic_sum += np.sum(values, axis=(0, -1))
            for name, dataset in bookkeeping.items():
                dataset[start:stop] = part[f"raw/{name}"][()]
        out.require_group("avg").create_dataset(
            "kinetic_spacetime",
            data=kinetic_sum / total_sources / nt,
        )
        out.flush()
    os.replace(temp_path, final_path)
    return str(final_path)


__all__ = ["RingedQuark1pt", "finalize_ringed_quark_1pt_shards"]
