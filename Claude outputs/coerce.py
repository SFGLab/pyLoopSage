"""Convert raw strings (ini / CLI) into typed values."""
import re

_NUM = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z]*)\s*$")
_TRUE, _FALSE = {"true", "1", "y", "yes", "on"}, {"false", "0", "n", "no", "off"}


def _number(value, unit_expected):
    """Parse '100', '100 femtosecond', '0.1 ps' -> float in the expected unit."""
    if isinstance(value, (int, float)):
        return float(value)
    m = _NUM.match(str(value))
    if not m:
        raise ValueError(f"cannot read a number from {value!r}")
    x, unit = float(m.group(1)), m.group(2).lower()
    if unit_expected == "fs" and unit in ("ps", "picosecond"):
        x *= 1000.0
    return x


def _items(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [s.strip() for s in str(value).strip("[]").split(",") if s.strip()]


def coerce(param, value):
    """Typed value for a Param; None for empty input."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    k = param.kind
    if k == "bool":
        s = str(value).strip().lower()
        if s in _TRUE: return True
        if s in _FALSE: return False
        raise ValueError(f"{param.name}: cannot read a boolean from {value!r}")
    if k == "str":
        return str(value).strip()
    if k == "int":
        return int(round(_number(value, param.unit)))
    if k == "float":
        return _number(value, param.unit)
    if k == "floats":
        return [float(x) for x in _items(value)]
    if k == "strs":
        return [str(x) for x in _items(value)]
    raise ValueError(f"unknown kind {k}")
