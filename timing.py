import numba
numba.config.CACHE_DEBUG = 1
from time import time, perf_counter

import rustworkx as rx
from qcio import Structure
from spyrmsd.rmsd import symmrmsd

from qcinf import smiles_to_structure
from qcinf.algorithms.geometry import rmsd

from qcinf._backends.utils import compute_rmsd, kabsch, qcp_rotation, kabsch_numba
from qcinf._backends.utils2 import qcp_rmsd_numba


def benchmark_snap(s1, s2, align=True, factor="pendant", **kwargs):
    start = time()
    s_rmsd, perm = snap_rmsd(s1, s2, align=align, factor=factor, **kwargs)
    snap_time = time() - start
    print(f"{s_rmsd:.5f}", perm)
    print(f"snapRMSD time: {snap_time:.4f} s")
    return s_rmsd, perm, snap_time

def benchmark_rdkit(s1, s2):
    start = time()
    rdkit_rmsd = rmsd(s1, s2, backend="rdkit")
    rdkit_time = time() - start
    print(f"{rdkit_rmsd:.5f}")
    print(f"RDKit RMSD time: {rdkit_time:.4f} s")
    return rdkit_rmsd, rdkit_time

def benchmark_symmrmsd(s1, s2, align=True):
    start = time()
    spyrmsd_rmsd = symmrmsd(
        s1.geometry,
        s2.geometry,
        s1.symbols,
        s2.symbols,
        s1.adjacency_matrix,
        s2.adjacency_matrix,
        center=True,
        minimize=align,
        cache=False,
    )
    spyrmsd_rmsd_time = time() - start
    print(f"{spyrmsd_rmsd:.5f}")
    print(f"spyrmsd RMSD time: {spyrmsd_rmsd_time:.4f} s")
    return spyrmsd_rmsd

def bench_pair(s1, s2, n=10000):
    A = s1.geometry
    B = s2.geometry
    print(A.dtype, A.flags.c_contiguous)
    print(B.dtype, B.flags.c_contiguous)

    # Warmup
    for _ in range(0):
        kabsch(A, B)
        qcp_rotation(A, B)
        kabsch_numba(A, B)
        compute_rmsd(A, B)
        qcp_rmsd_numba(A, B)

    # Kabsch
    t0 = perf_counter()
    for _ in range(n):
        kabsch(A, B)
    t_kabsch = perf_counter() - t0

    # QCP
    t0 = perf_counter()
    for _ in range(n):
        qcp_rotation(A, B)
    t_qcp = perf_counter() - t0

    # Kabsch Numba
    t0 = perf_counter()
    for _ in range(n):
        kabsch_numba(A, B)
    t_kabsch_numba = perf_counter() - t0

    # Regular RMSD
    t0 = perf_counter()
    for _ in range(n):
        rmsd_val = compute_rmsd(A, B)
    t_rmsd = perf_counter() - t0

    # RMSD Numba
    t0 = perf_counter()
    for _ in range(n):
        qcp_rmsd_val = qcp_rmsd_numba(A, B)

    t_rmsd_numba = perf_counter() - t0

    print(f"N = {A.shape[0]}")
    print(f"  kabsch:     {t_kabsch:.6f} s total, {t_kabsch / n:.3e} s/call")
    print(f"  qcp_rot:    {t_qcp:.6f} s total, {t_qcp / n:.3e} s/call")
    print(f"  kabsch_numba: {t_kabsch_numba:.6f} s total, {t_kabsch_numba / n:.3e} s/call")  # fmt: skip
    print(f"  rmsd:       {t_rmsd:.6f} s total, {t_rmsd / n:.3e} s/call")
    print(f"  rmsd_numba: {t_rmsd_numba:.6f} s total, {t_rmsd_numba / n:.3e} s/call")  # fmt: skip
    print(f"  rmsd_val: {rmsd_val:.6f}, qcp_rmsd_val: {qcp_rmsd_val:.6f}")

smiles = "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C"
# smiles = "FC(F)(F)C1=CC(=CC(NC(=O)NC2=CC(=CC(=C2)C(F)(F)F)C(F)(F)F)=C1)C(F)(F)F"
# smiles = "CCCCC"
s1 = smiles_to_structure(smiles)
s2 = smiles_to_structure(smiles)

# R, cA, cB = kabsch(s1.geometry, s2.geometry)
# R_q, cA_q, cB_q = qcp_rotation(s1.geometry, s2.geometry)
bench_pair(s1, s2, n=1)

# s2 = randomly_reorder_structure(s2)
# s1.save("s1.json")
# s2.save("s2.json")
# s1 = Structure.open("s1.json")
# s2 = Structure.open("s2.json")

# g1 = rx.PyGraph.from_adjacency_matrix(s1.adjacency_matrix)
# g1 = _struct_to_rustworkx_graph(s1)
# g2 = _struct_to_rustworkx_graph(s2)
# assert rx.is_isomorphic(g1, g2), "Rustworkx says not isomorphic!"

# batches = _pendant_factor(s1)
# s_rmsd, perm, snap_time = benchmark_snap(s1, s2, align=True, factor="pendant")
# s_rmsd, perm, snap_time_unfact = benchmark_snap(s1, s2, align=True, factor_depth=0)
# rdkit_rmsd, rdkit_time = benchmark_rdkit(s1, s2)
# spyrmsd_rmsd = benchmark_symmrmsd(s1, s2, align=True)
# raw = compute_rmsd(s1.geometry, s2.geometry, align=False)
# print(raw)
# Compare s1_geom to s1.geometry to ensure no mutation
# print(f"Speedup over RDKit: {rdkit_time / snap_time:.2f}x")

# import cProfile, pstats, io

# pr = cProfile.Profile()
# pr.enable()

# # the workload
# s_rmsd, perm = snap_rmsd(s1, s2, align=True, factor="pendant", factor_depth=1)

# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")  # or "tottime"
# ps.print_stats(30)  # top 30
# print(s.getvalue())
# pr.dump_stats("profile.out")
