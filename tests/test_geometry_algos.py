import numpy as np
import pytest
from qcconst import constants
from qcconst.constants import ANGSTROM_TO_BOHR
from qcio import ConformerSearchResults, Structure

from qcinf import align, filter_conformers_indices
from qcinf.algorithms.geometry import _ALIGN_BACKEND_MAP


@pytest.mark.parametrize(
    "backend", _ALIGN_BACKEND_MAP.keys(), ids=_ALIGN_BACKEND_MAP.keys()
)
def test_align_identical_structures(backend):
    """Test that aligning identical structures results in no change."""
    symbols = ["N", "H", "H", "H"]
    geometry = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.9377, 0.0, 0.0],
                [-0.3126, 0.8892, 0.0],
                [-0.3126, -0.4446, 0.7697],
            ]
        )
        * ANGSTROM_TO_BOHR
    )  # Ammonia molecule

    struct = Structure(symbols=symbols, geometry=geometry)
    refstruct = Structure(symbols=symbols, geometry=geometry)

    aligned_struct, rmsd_val = align(struct, refstruct, symmetry=False, backend=backend)

    assert np.allclose(aligned_struct.geometry, geometry), (
        "Aligned geometry should be the same as original"
    )


@pytest.mark.parametrize(
    "backend", _ALIGN_BACKEND_MAP.keys(), ids=_ALIGN_BACKEND_MAP.keys()
)
def test_align_rotated_structure(backend):
    """Test aligning a rotated structure to the reference structure."""
    symbols = ["O", "H", "H"]
    geometry = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.9584, 0.0, 0.0],
                [-0.2396, 0.9270, 0.0],
            ]
        )
        * ANGSTROM_TO_BOHR
    )

    # Rotate struct by 90 degrees around the y-axis
    theta = np.pi / 2  # 90 degrees
    rotation_matrix = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    rotated_geometry = geometry @ rotation_matrix.T

    struct = Structure(symbols=symbols, geometry=rotated_geometry)
    refstruct = Structure(symbols=symbols, geometry=geometry)

    aligned_struct, rmsd = align(struct, refstruct, symmetry=False, backend=backend)

    assert np.allclose(aligned_struct.geometry, geometry, atol=1e-6), (
        "Aligned geometry should match the reference"
    )


@pytest.mark.parametrize(
    "backend", _ALIGN_BACKEND_MAP.keys(), ids=_ALIGN_BACKEND_MAP.keys()
)
def test_align_raises_value_error_reorder_different_atom_count_if_symmetry(backend):
    """Test that an error is raised when structures have different atom counts."""
    symbols1 = ["C", "H", "H", "H", "H"]
    geometry1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.19052746, 1.19052746, 1.19052746],
            [-1.19052746, -1.19052746, 1.19052746],
            [-1.19052746, 1.19052746, -1.19052746],
            [1.19052746, -1.19052746, -1.19052746],
        ]
    )

    symbols2 = ["C", "H", "H", "H"]
    geometry2 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.19052746, 1.19052746, 1.19052746],
            [-1.19052746, -1.19052746, 1.19052746],
            [-1.19052746, 1.19052746, -1.19052746],
        ]
    )

    struct1 = Structure(symbols=symbols1, geometry=geometry1)
    struct2 = Structure(symbols=symbols2, geometry=geometry2)
    with pytest.raises(ValueError):
        align(struct2, struct1, backend=backend)


def test_align_large_molecule():
    """Test aligning larger molecules."""
    symbols = ["C"] * 10
    geometry = np.random.rand(10, 3) * ANGSTROM_TO_BOHR

    struct1 = Structure(symbols=symbols, geometry=geometry)
    struct2 = Structure(symbols=symbols, geometry=geometry + 0.1)  # Slightly shifted

    aligned_struct, calculated_rmsd = align(struct1, struct2, symmetry=False)
    assert calculated_rmsd < 0.2, "RMSD should be low for slightly shifted structures"


# Note: This test is slow since it uses RDKit's Hueckel method to determine connectivity
# and then uses the Hungarian algorithm to consider symmetries. If needed perhaps we can
# modify this test to run more quickly by changing the connectivity determination method
# and/or not considering symmetries.
def test_conformers_filtered(test_data_dir):
    # Catalyst/Na+ conformer search
    csr = ConformerSearchResults.open(test_data_dir / "conf_search.json")
    keep_indices = filter_conformers_indices(
        csr.conformers, backend="rdkit", threshold=0.47 * constants.ANGSTROM_TO_BOHR
    )
    assert len(keep_indices) == 6
    assert keep_indices == [0, 5, 7, 8, 9, 10]
    # for i, conf in enumerate(keep_indices.conformers):
    #     assert conf == csr.conformers[selected[i]]
    #     assert (
    #         keep_indices.conformer_energies[i]
    #         == csr.conformer_energies_relative[selected[i]]
    #     )
