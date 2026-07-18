"""Connected proton C2 and nonlocal-line helpers for qTMD/PDF production."""

from pyquda_measurement_utils.fermion_bilinear_basis import (
    GAMMA_LABELS,
)
from pyquda_measurement_utils.io_corr import save_proton_c2pt_hdf5
from pyquda_measurement_utils.proton_utils_vibe_develop import contract_proton_c2

class proton_TMD():
    def __init__(self, parameters):
        self.pilist = parameters["p_2pt"]  # 2pt momentum
        self.width = parameters["width"] # Gaussian smearing width
        self.boost_out = parameters["boost_out"] # Sink-propagator boost smearing
        
    #! PyQUDA: contract 2pt TMD
    def contract_2pt_TMD(
        self, latt_info, prop_f, phases, tag, interpolator="5", attrs=None
    ):
        """Contract and write proton C2 through the shared calculation kernel."""
        corr_collect = contract_proton_c2(
            latt_info,
            prop_f,
            phases,
            interpolator=interpolator,
            sink_smearing=True,
            smearing_width=self.width,
            smearing_boost=self.boost_out,
        )
        if latt_info.mpi_rank == 0:
            save_proton_c2pt_hdf5(
                corr_collect, tag, list(GAMMA_LABELS), self.pilist, attrs=attrs
            )
        return corr_collect
