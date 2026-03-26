'''
Modified by Jinchen He, 2025-11-21.
Refactored for MPI support using pyquda_utils.fft by Gemini.
Optimized: Removed U_trafo dependency (Assuming Identity Gauge).

This module implements boosted smearing in PyQUDA.
It supports MPI parallelization by leveraging PyQUDA's internal distributed FFT utils.

momentum smearing: add a momentum phase to the gauge, then do the gaussian smearing;
boosted smearing: apply a gauge-covariant Gaussian convolution with an injected momentum: the source is first rotated into the fixed gauge frame, Fourier-transformed, multiplied by the momentum-shifted Gaussian kernel in momentum space, inverse-transformed back to position space, and finally rotated back with the hermitian conjugate of the gauge transformation.
'''

from typing import Sequence
from math import pi
import numpy as np

# PyQUDA imports
from pyquda.field import Ns, Nc
from pyquda.field import LatticeInfo, LatticeFermion, LatticePropagator, LatticeComplex

# Import PyQUDA's distributed FFT
try:
    from pyquda_utils.fft import fft, ifft
except ImportError:
    raise ImportError("Could not import 'fft' from 'pyquda_utils'. Please ensure PyQUDA is installed correctly.")

from pyquda_comm.array import arrayExp
from pyquda_utils.phase_v2 import DistancePhase

def _build_kernel_realspace_distributed(latt_info: LatticeInfo, w: float, boost: Sequence[float]):
    """
    build the distributed real space Gaussian kernel.
    return a LatticeComplex object (Checkerboard layout), which can be directly passed to fft.
    """
    r = DistancePhase(latt_info).getPhase([0, 0, 0, 0])
    rx, ry, rz = r[0], r[1], r[2]
    Gx, Gy, Gz, Gt = latt_info.global_size
    
    kx, ky, kz = boost

    # calculate the exponential part
    real = (-0.5 / (w * w)) * (rx.data**2 + ry.data**2 + rz.data**2)
    imag = 2 * pi * ((kx / Gx) * rx.data + (ky / Gy) * ry.data + (kz / Gz) * rz.data)

    kernel_field = LatticeComplex(latt_info, arrayExp(real + 1j * imag, r.backend))
    
    return kernel_field

def _boosted_smearing_fermion(src: LatticeFermion, *, w: float, boost: Sequence[float]):
    """
    Core implementation of boosted smearing for a single fermion.
    Optimized: Assumes Identity Gauge (No U_trafo input).
    """
    latt_info: LatticeInfo = src.latt_info

    # ---------------------------------------------------------
    # 1. Forward FFT (Distributed)
    # ---------------------------------------------------------
    # because U=Identity, so we don't need to do src.lexico() -> einsum -> evenodd()
    # directly do FFT on LatticeFermion
    psi_p = fft(src, fft3d=True, backend="cupy" if src.backend == "cupy" else "numpy")

    # ---------------------------------------------------------
    # 2. Apply Momentum Space Kernel
    # ---------------------------------------------------------
    K_xyz = _build_kernel_realspace_distributed(latt_info, w, boost)
    K_p = fft(K_xyz, fft3d=True, backend="cupy" if src.backend == "cupy" else "numpy")

    # multiply in momentum space: psi(k) * K(k)
    psi_p.data = psi_p.data * K_p.data[..., None, None]

    # ---------------------------------------------------------
    # 3. Inverse FFT (Distributed)
    # ---------------------------------------------------------
    psi_smeared = ifft(psi_p, fft3d=True, backend="cupy" if src.backend == "cupy" else "numpy")

    # ---------------------------------------------------------
    # 4. Result
    # ---------------------------------------------------------
    # because U=Identity, we don't need to do the inverse Gauge Rotation
    # psi_smeared is already the final result
    
    return psi_smeared

# ---------- public API ----------
def boosted_smearing(
    src,
    *,
    w: float,
    boost: Sequence[float],
):
    if isinstance(src, LatticeFermion):
        return _boosted_smearing_fermion(src, w=w, boost=boost)
    if isinstance(src, LatticePropagator):
        out = LatticePropagator(src.latt_info)
        for s in range(Ns):
            for c in range(Nc):
                # pass in a single fermion
                f_sm = _boosted_smearing_fermion(src.getFermion(s, c), w=w, boost=boost)
                out.setFermion(f_sm, s, c)
        return out
    raise TypeError(f"boosted_smearing: unsupported src type: {type(src)}")
