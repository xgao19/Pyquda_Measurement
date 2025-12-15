import os
import types

try:
    import torch
    import opt_einsum as oe
except Exception:
    torch = None
    oe = None

if torch is not None and oe is not None:
    _orig_contract = oe.contract

    def _is_xpu_complex_tensor(x):
        return isinstance(x, torch.Tensor) and x.is_complex() and x.device.type == "xpu"

    def _to_cpu(x):
        return x.to("cpu") if isinstance(x, torch.Tensor) else x

    def _maybe_to_xpu(x, like_operands):
        if isinstance(x, torch.Tensor):
            # send back to xpu if any input was on xpu
            if any(isinstance(t, torch.Tensor) and t.device.type == "xpu" for t in like_operands):
                return x.to("xpu")
        return x

    def _wrapped_contract(*args, **kwargs):
        # Expect signature: (subscripts, *operands, **kwargs)
        if not args:
            return _orig_contract(*args, **kwargs)
        subscripts = args[0]
        operands = args[1:]
        if any(_is_xpu_complex_tensor(t) for t in operands):
            cpu_operands = tuple(_to_cpu(t) for t in operands)
            out = _orig_contract(subscripts, *cpu_operands, **kwargs)
            return _maybe_to_xpu(out, operands)
        return _orig_contract(*args, **kwargs)

    # Apply monkeypatch
    oe.contract = _wrapped_contract
