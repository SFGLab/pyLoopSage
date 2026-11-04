"""Human-readable formatting with explicit units."""


def fmt_bp(bp: float) -> str:
    bp = float(bp)
    if abs(bp) >= 1e6:
        return f"{bp / 1e6:.2f} Mb"
    if abs(bp) >= 1e3:
        return f"{bp / 1e3:.1f} kb"
    return f"{bp:.0f} bp"


def fmt_num(x, digits: int = 4) -> str:
    """Compact number: integers stay integers, floats get significant digits."""
    if isinstance(x, (int,)) or (isinstance(x, float) and x.is_integer() and abs(x) < 1e9):
        return f"{int(x):,}"
    return f"{x:.{digits}g}"


def fmt_duration(seconds: float) -> str:
    s = float(seconds)
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{int(s // 60)} min {int(s % 60)} s"
    return f"{int(s // 3600)} h {int(s % 3600 // 60)} min"
