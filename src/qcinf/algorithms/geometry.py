"""Top-level functions for geometry-related algorithms."""

from collections.abc import Callable

from qcio import LengthUnit, Structure

import qcinf._backends.qcinf as qcinf_be
from qcinf._backends import rdkit

_RMSD_BACKEND_MAP: dict[str, Callable[..., float]] = {
    "qcinf": qcinf_be._rmsd,
    "rdkit": rdkit._rmsd_rdkit,
}


def rmsd(
    struct1: Structure,
    struct2: Structure,
    *,
    backend: str = "qcinf",
    length_unit: str = LengthUnit.BOHR,
    **kwargs,
) -> float:
    """Calculate the root mean square deviation (RMSD) between two structures.

    Args:
        struct1: The first structure.
        struct2: The second structure.
        backend: The backend to use for the RMSD calculation. Defaults to 'qcinf'.
        length_unit: The length unit to use for the RMSD calculation. Defaults to
            LengthUnit.BOHR. Other options include LengthUnit.ANGSTROM.
        **kwargs: Backend-specific additional keywords to pass to the RMSD calculation
            function. This can include options like 'symmetry' for symmetry-based RMSD
            calculations. The specific options depend on the backend used. See
            [qcinf._backends.<backend>._rmsd_<backend>] for details.

    Returns:
        The RMSD value between the two structures.

    Raises:
        ValueError: If the backend is not one of the supported options.
    """
    try:
        fn = _RMSD_BACKEND_MAP[backend.lower()]
    except KeyError as e:
        raise ValueError(
            f"Unknown backend '{backend}'.  Known: {list(_RMSD_BACKEND_MAP)}"
        ) from e
    return fn(struct1, struct2, length_unit=length_unit, **kwargs)


_ALIGN_BACKEND_MAP: dict[str, Callable[..., tuple[Structure, float]]] = {
    "qcinf": qcinf_be._align,
    "rdkit": rdkit._align_rdkit,
}


def align(
    struct: Structure,
    refstruct: Structure,
    *,
    symmetry: bool = True,
    backend: str = "qcinf",
    **kwargs,
) -> tuple[Structure, float]:
    """Align a structure to a reference structure.

    Args:
        struct: The structure to align.
        refstruct: The reference structure to align against.
        symmetry: Whether to consider symmetry in the alignment. Defaults to True.
        backend: The backend to use for the alignment. Can be 'rdkit'.
        **kwargs: Additional keyword arguments to pass to the alignment function.

    Returns:
        A tuple containing the RMSD and the aligned structure.
    """
    try:
        fn = _ALIGN_BACKEND_MAP[backend.lower()]
    except KeyError as e:
        raise ValueError(
            f"Unknown backend '{backend}'.  Known: {list(_ALIGN_BACKEND_MAP)}"
        ) from e
    return fn(struct, refstruct, symmetry=symmetry, **kwargs)


def filter_conformers_indices(
    conformers: list[Structure],
    threshold: float = 1.0,
    backend: str = "rdkit",
    **rmsd_kwargs,
) -> list[int]:
    """Filter conformers based on RMSD threshold.

    Args:
        conformers: A list of Structure objects to filter
        threshold: The RMSD threshold for filtering in Bohr. Defaults to 1.0 Bohr
            (0.53 Angstrom).
        backend: The backend to use for the RMSD calculation. Can be 'rdkit'.
        **rmsd_kwargs: Additional keyword arguments to pass to the RMSD calculation
            function. This can include options like 'symmetry' for symmetry-based RMSD
            calculations. The specific options depend on the backend used. See
            [qcinf._backends.<backend>._rmsd_<backend>] for details.

    Returns:
        List of conformers indices that meet the RMSD threshold.
    """
    filtered = set()
    for i in range(len(conformers)):
        if i not in filtered:
            for j in range(i + 1, len(conformers)):
                if (
                    rmsd(
                        conformers[i],
                        conformers[j],
                        backend=backend,
                        **rmsd_kwargs,
                    )
                    < threshold
                ):
                    filtered.add(j)

    return [i for i in range(len(conformers)) if i not in filtered]


def filter_conformers(
    conformers: list[Structure],
    threshold: float = 1.0,
    backend: str = "rdkit",
    **rmsd_kwargs,
) -> list[Structure]:
    """Filter conformers based on RMSD threshold.

    Args:
        conformers: A list of Structure objects to filter
        threshold: The RMSD threshold for filtering in Bohr. Defaults to 1.0 Bohr
            (0.53 Angstrom).
        backend: The backend to use for the RMSD calculation. Can be 'rdkit'.
        **rmsd_kwargs: Additional keyword arguments to pass to the RMSD calculation
            function. This can include options like 'symmetry' for symmetry-based RMSD
            calculations. The specific options depend on the backend used. See
            [qcinf._backends.<backend>._rmsd_<backend>] for details.

    Returns:
        A list of filtered conformers that meet the RMSD threshold.
    """
    keep_indices = filter_conformers_indices(
        conformers, threshold=threshold, backend=backend, **rmsd_kwargs
    )
    return [conformers[i] for i in keep_indices]
