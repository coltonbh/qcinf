from time import perf_counter

import numba

numba.config.CACHE_DEBUG = 1
from time import perf_counter

from qcinf import smiles_to_structure
from qcinf._backends.utils import kabsch_numba

smiles = "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C"
s1 = smiles_to_structure(smiles)
s2 = smiles_to_structure(smiles)

A = s1.geometry
B = s2.geometry

t0 = perf_counter()
kabsch_numba(A, B)
t1 = perf_counter()
print("first call:", t1 - t0)

t0 = perf_counter()
kabsch_numba(A, B)
t1 = perf_counter()
print("second call:", t1 - t0)
