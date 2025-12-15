"""Required functions for optimized contractions of numpy arrays using pytorch."""

from opt_einsum.helpers import has_array_interface
from opt_einsum.parser import convert_to_valid_einsum_chars
from opt_einsum.sharing import to_backend_cache_wrap

__all__ = [
    "transpose",
    "einsum",
    "tensordot",
    "to_torch",
    "build_expression",
    "evaluate_constants",
]

_TORCH_DEVICE = None
_TORCH_HAS_TENSORDOT = None

_torch_symbols_base = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _get_torch_and_device():
    global _TORCH_DEVICE
    global _TORCH_HAS_TENSORDOT

    if _TORCH_DEVICE is None:
        import torch  # type: ignore

        device = "cpu"
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            elif torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"
        _TORCH_DEVICE = torch, device
        _TORCH_HAS_TENSORDOT = hasattr(torch, "tensordot")

    return _TORCH_DEVICE


def transpose(a, axes):
    """Normal torch transpose is only valid for 2D matrices."""
    return a.permute(*axes)


def einsum(equation, *operands, **kwargs):
    """Variadic version of torch.einsum to match numpy api."""
    # rename symbols to support PyTorch 0.4.1 and earlier,
    # which allow only symbols a-z.
    equation = convert_to_valid_einsum_chars(equation)

    torch, _ = _get_torch_and_device()
    # Ensure numpy operands are converted
    ops = tuple(to_torch(op) for op in operands)
    # Fallback: oneDNN on XPU currently lacks complex matmul; compute on CPU
    try:
        any_xpu = False
        any_complex = False
        target_device = None
        for op in ops:
            if hasattr(op, "device"):
                if getattr(op.device, "type", None) == "xpu":
                    any_xpu = True
                    if target_device is None:
                        target_device = op.device
                if torch.is_complex(op):
                    any_complex = True
        if any_xpu and any_complex:
            ops_cpu = tuple(op.to("cpu") if hasattr(op, "to") else op for op in ops)
            out_cpu = torch.einsum(equation, ops_cpu)
            if target_device is not None:
                return out_cpu.to(target_device)
            return out_cpu
    except Exception:
        # If detection or transfer fails, fall through to default behavior
        pass

    return torch.einsum(equation, ops)


def tensordot(x, y, axes=2):
    """Simple translation of tensordot syntax to einsum."""
    torch, _ = _get_torch_and_device()
    # Convert numpy inputs
    x = to_torch(x)
    y = to_torch(y)

    if _TORCH_HAS_TENSORDOT:
        # CPU fallback for complex on XPU
        try:
            x_dev = getattr(getattr(x, "device", None), "type", None)
            y_dev = getattr(getattr(y, "device", None), "type", None)
            target_device = x.device if x_dev in ("xpu", "cuda") else (y.device if y_dev in ("xpu", "cuda") else None)
            if (x_dev == "xpu" or y_dev == "xpu") and (torch.is_complex(x) or torch.is_complex(y)):
                out_cpu = torch.tensordot(x.to("cpu"), y.to("cpu"), dims=axes)
                if target_device is not None:
                    return out_cpu.to(target_device)
                return out_cpu
        except Exception:
            pass
        return torch.tensordot(x, y, dims=axes)

    xnd = x.ndimension()
    ynd = y.ndimension()

    # convert int argument to (list[int], list[int])
    if isinstance(axes, int):
        axes = range(xnd - axes, xnd), range(axes)

    # convert (int, int) to (list[int], list[int])
    if isinstance(axes[0], int):
        axes = (axes[0],), axes[1]
    if isinstance(axes[1], int):
        axes = axes[0], (axes[1],)

    # initialize empty indices
    x_ix = [None] * xnd
    y_ix = [None] * ynd
    out_ix = []

    # fill in repeated indices
    available_ix = iter(_torch_symbols_base)
    for ax1, ax2 in zip(*axes):
        repeat = next(available_ix)
        x_ix[ax1] = repeat
        y_ix[ax2] = repeat

    # fill in the rest, and maintain output order
    for i in range(xnd):
        if x_ix[i] is None:
            leave = next(available_ix)
            x_ix[i] = leave
            out_ix.append(leave)
    for i in range(ynd):
        if y_ix[i] is None:
            leave = next(available_ix)
            y_ix[i] = leave
            out_ix.append(leave)

    # form full string and contract!
    einsum_str = "{},{}->{}".format(*map("".join, (x_ix, y_ix, out_ix)))
    return einsum(einsum_str, x, y)


@to_backend_cache_wrap
def to_torch(array):
    torch, device = _get_torch_and_device()

    if has_array_interface(array):
        return torch.from_numpy(array).to(device)

    return array


def build_expression(_, expr):  # pragma: no cover
    """Build a torch function based on ``arrays`` and ``expr``."""

    def torch_contract(*arrays):
        torch_arrays = [to_torch(x) for x in arrays]
        torch_out = expr._contract(torch_arrays, backend="torch")

        if torch_out.device.type == "cpu":
            return torch_out.numpy()

        return torch_out.cpu().numpy()

    return torch_contract


def evaluate_constants(const_arrays, expr):
    """Convert constant arguments to torch, and perform any possible constant
    contractions.
    """
    const_arrays = [to_torch(x) for x in const_arrays]
    return expr(*const_arrays, backend="torch", evaluate_constants=True)
