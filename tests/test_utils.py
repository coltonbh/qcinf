import numpy as np

from qcinf._backends.utils import rotate_structure


def test_rotate_structure(water):
    """Test that the rotate_structure function correctly rotates a structure."""

    # Rotate by 90 degrees around the z-axis
    rotated_struct = rotate_structure(water, "z", 90)

    # Check if the rotation is correct
    expected_geometry = np.array(
        [
            [-0.01939466, 0.0253397, -0.00696322],
            [-1.84438441, 0.22889176, 0.16251426],
            [0.62610794, 1.41760224, -1.02954938],
        ]
    )
    assert np.allclose(rotated_struct.geometry, expected_geometry), (
        "Rotated geometry does not match expected geometry"
    )
