import numpy as np
import pytest

from qcinf._backends.qcinf import _rmsd
from qcinf._backends.utils import rotate_structure


def test_rmsd_identity(water):
    assert _rmsd(water, water) == pytest.approx(0.0, abs=1e-6)


def test_rmsd_rotated_structures(water):
    """Test that RMSD between rotated structures is zero when aligned."""
    # Rotate by 180 degrees around the z-axis
    rotated_water = rotate_structure(water, "z", 180)

    # Aligned by atom index (which match in this case)
    calculated_rmsd_no_align = _rmsd(water, rotated_water, align=True)
    assert np.isclose(calculated_rmsd_no_align, 0.0, atol=1e-6), (
        "RMSD should be zero after alignment"
    )
