"""N/W output-basis metadata, shared by readers and dependency-light converters."""


def normalize_evpa_zero(value):
    if isinstance(value, bytes):
        value = value.decode()
    if value == "camera":  # Explicit label used by pre-N/W KPolaris files.
        return "W"
    if value not in ("N", "W"):
        raise ValueError(f"unknown evpa_0={value!r}; expected N, W, or legacy camera")
    return value


def evpa_zero(*attribute_sets):
    """Read KPolaris metadata; old files without this attribute used camera/W.

    Root and selected-frequency declarations must agree. Do not call this
    legacy fallback on an external code's file with an unknown convention.
    """
    declared = {normalize_evpa_zero(attrs["evpa_0"])
                for attrs in attribute_sets if "evpa_0" in attrs}
    if len(declared) > 1:
        raise ValueError("conflicting evpa_0 declarations in the input file")
    return next(iter(declared), "W")


def qu_basis_sign(source, target):
    """Passive 90-degree basis change; I and V remain unchanged."""
    return 1 if normalize_evpa_zero(source) == normalize_evpa_zero(target) else -1
