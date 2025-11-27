import numpy as np
import pytest
from qcio import Structure

from qcinf._backends.utils import rotate_structure

pytest.importorskip("rdkit")  # tries `import rdkit` skips tests if not installed


from qcinf._backends.rdkit import (
    _align_rdkit,
    _rmsd_rdkit,
    _smiles_to_structure_rdkit,
    _structure_to_smiles_rdkit,
)


def test_rmsd_identity(water):
    assert _rmsd_rdkit(water, water) == pytest.approx(0.0, abs=1e-6)


def test_rmsd_alignment_happens_before_rmsd(water):
    water2 = rotate_structure(water, "z", 90.0)
    assert _rmsd_rdkit(water, water2, symmetry=False) == pytest.approx(0.0, abs=1e-6)


def test_smiles_to_structure():
    struct = _smiles_to_structure_rdkit("OCC", force_field="UFF")
    assert struct.symbols == ["O", "C", "C", "H", "H", "H", "H", "H", "H"]
    assert struct.charge == 0
    assert struct.multiplicity == 1
    assert struct.identifiers.smiles == "OCC"
    assert struct.identifiers.canonical_smiles == "CCO"
    assert struct.connectivity == [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (0, 3, 1.0),
        (1, 4, 1.0),
        (1, 5, 1.0),
        (2, 6, 1.0),
        (2, 7, 1.0),
        (2, 8, 1.0),
    ]


def test_smiles_to_structure_aromaticity_connectivity():
    struct = _smiles_to_structure_rdkit("c1ccccc1")  # benzene
    assert struct.connectivity == [
        (0, 1, 1.5),
        (1, 2, 1.5),
        (2, 3, 1.5),
        (3, 4, 1.5),
        (4, 5, 1.5),
        (5, 0, 1.5),
        (0, 6, 1.0),
        (1, 7, 1.0),
        (2, 8, 1.0),
        (3, 9, 1.0),
        (4, 10, 1.0),
        (5, 11, 1.0),
    ]


def test_smiles_to_structure_charges():
    # Check Charge
    struct = _smiles_to_structure_rdkit("[O-]CC")
    assert struct.charge == -1


def test_smiles_to_structure_multiplicity():
    # Check manual multiplicity
    struct = _smiles_to_structure_rdkit("[O-]CC", multiplicity=3)
    assert struct.charge == -1
    assert struct.multiplicity == 3


def test_smiles_charges_round_trip():
    """Test that SMILES with charges are handled correctly."""
    s = _smiles_to_structure_rdkit("CC[O-]")
    assert s.charge == -1
    # Using robust method
    assert _structure_to_smiles_rdkit(s) == "CC[O-]"


def test_structure_to_smiles_hydrogens(water):
    smiles = _structure_to_smiles_rdkit(water)
    assert smiles == "O"
    smiles = _structure_to_smiles_rdkit(water, hydrogens=True)
    assert smiles == "[H]O[H]"


def test_rmsd_with_numthreads():
    """Test rdkit RMSD calculation with multiple threads."""
    symbols = ["O", "H", "H"]
    geometry = np.array(
        [[0.0, 0.0, 0.0], [1.43354624, 0.0, 0.95298889], [-1.43354624, 0.0, 0.95298889]]
    )

    struct1 = Structure(symbols=symbols, geometry=geometry)
    struct2 = Structure(symbols=symbols, geometry=geometry)

    rmsd_single_thread = _rmsd_rdkit(struct1, struct2, symmetry=True, numthreads=1)
    rmsd_multi_thread = _rmsd_rdkit(struct1, struct2, symmetry=True, numthreads=4)

    assert np.isclose(rmsd_single_thread, 0.0, atol=1e-6), (
        "RMSD should be zero with single thread"
    )
    assert np.isclose(rmsd_multi_thread, 0.0, atol=1e-6), (
        "RMSD should be zero with multiple threads"
    )


def test_align_incorrect_atom_mapping():
    """Test that an error is raised when atom mapping fails.

    NOTE: May want to update this to an internal qcinf error in the future
        shared by all backends.
    """
    symbols1 = ["C", "H", "H", "H", "H"]
    symbols2 = ["O", "H", "H", "H", "H"]
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.19052746, 1.19052746, 1.19052746],
            [-1.19052746, -1.19052746, 1.19052746],
            [-1.19052746, 1.19052746, -1.19052746],
            [1.19052746, -1.19052746, -1.19052746],
        ]
    )

    struct1 = Structure(symbols=symbols1, geometry=geometry)
    struct2 = Structure(symbols=symbols2, geometry=geometry)

    with pytest.raises(RuntimeError):  # Raised by RDKit
        _align_rdkit(struct1, struct2, symmetry=False)


def test_align_with_atom_symmetry():
    """Test aligning structures with atom reordering (symmetry=True) allowed.

    NOTE: At some point this should be a shared test for all backends when I have
        my own symmetry implementation.
    """
    symbols1 = ["C", "H", "H", "H", "Cl"]
    geometry1 = np.array(
        [
            [0.0, 0.0, 0.0],  # C
            [0.0, 0.0, 2.05980148],  # H
            [1.94018181, 0.0, -0.6865375],  # H
            [-0.96999642, -1.68034447, -0.6865375],  # H
            [-0.96999642, 1.68034447, -0.6865375],  # Cl
        ]
    )

    symbols2 = ["C", "Cl", "H", "H", "H"]
    geometry2 = np.array(
        [
            [0.0, 0.0, 0.0],  # C
            [-0.96999642, 1.68034447, -0.6865375],  # Cl
            [0.0, 0.0, 2.05980148],  # H
            [-0.96999642, -1.68034447, -0.6865375],  # H
            [1.94018181, 0.0, -0.6865375],  # H
        ]
    )

    struct1 = Structure(symbols=symbols1, geometry=geometry1)
    struct2 = Structure(symbols=symbols2, geometry=geometry2)

    # Without atom reordering
    aligned_struct_no_reorder, rmsd_no_reorder = _align_rdkit(
        struct1, struct2, symmetry=False
    )
    assert aligned_struct_no_reorder.symbols == struct1.symbols, (
        "Symbols should not be reordered"
    )
    assert rmsd_no_reorder > 0.1, "RMSD should be high without atom reordering"

    # With atom reordering
    aligned_struct_reorder, rmsd_reorder = _align_rdkit(struct1, struct2, symmetry=True)
    rmsd_reorder = _rmsd_rdkit(aligned_struct_reorder, struct2, symmetry=False)
    assert aligned_struct_reorder.symbols == struct2.symbols, (
        "Symbols should be reordered"
    )

    assert np.isclose(rmsd_reorder, 0.0, atol=1e-2), (
        "RMSD should be zero with atom reordering"
    )
