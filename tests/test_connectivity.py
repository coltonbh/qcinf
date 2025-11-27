from qcio import Structure

from qcinf import determine_connectivity


def test_determine_connectivity_water():
    """Test connectivity determination on a water molecule."""
    struct = Structure(symbols=["O", "H", "H"], geometry=[[0.0, 0.0, 0.0], [1.83, 0.0, 0.0], [0.0, 1.83, 0.0]])  # fmt: skip
    connectivity = determine_connectivity(struct)

    expected_connectivity = [(0, 1, 1.0), (0, 2, 1.0)]  # O-H bonds
    assert connectivity == expected_connectivity, (
        f"Expected {expected_connectivity}, got {connectivity}"
    )


def test_determine_connectivity_single_atom():
    """Test connectivity determination on a single atom (no bonds)."""
    struct = Structure(symbols=["He"], geometry=[[0.0, 0.0, 0.0]])
    connectivity = determine_connectivity(struct)
    assert connectivity == []


def test_determine_connectivity_unconnected():
    """Test connectivity determination on unconnected atoms."""
    struct = Structure(
        symbols=["He", "Ne"], geometry=[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
    )
    connectivity = determine_connectivity(struct)
    assert connectivity == []
