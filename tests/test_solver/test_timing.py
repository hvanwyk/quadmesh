import unittest
import time
import numpy as np
# Compatibility shim for NumPy >=1.20 where np.int was removed
if not hasattr(np, "int"):
    np.int = int

from assembler import Form, Kernel, Assembler
from mesh import QuadMesh
from fem import DofHandler, Basis, QuadFE
from function import Nodal, Constant
from solver import LinearSystem

try:
    from diagnostics import limit_math_threads, current_threadpools
except Exception:
    # Fallbacks if diagnostics helpers are unavailable
    def limit_math_threads(n):
        class _Noop:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
        return _Noop()
    def current_threadpools():
        return []


class TestSolverTiming(unittest.TestCase):
    def _build_poisson_2d(self, resolution=(32, 32)):
        mesh = QuadMesh(resolution=resolution)

        # Mark left/right boundaries for Dirichlet constraints
        bm_left = lambda x, dummy: np.abs(x) < 1e-9
        bm_right = lambda x, dummy: np.abs(1 - x) < 1e-9
        mesh.mark_region('left', bm_left, entity_type='half_edge')
        mesh.mark_region('right', bm_right, entity_type='half_edge')

        element = QuadFE(2, 'Q1')
        dh = DofHandler(mesh, element)
        dh.distribute_dofs()

        u = Basis(dh, 'u')
        ux = Basis(dh, 'ux')
        uy = Basis(dh, 'uy')

        # Bilinear form for Poisson: ∫ (ux*ux + uy*uy) dx, zero RHS
        one = Constant(1)
        zero = Constant(0)
        ax = Form(kernel=Kernel(one), trial=ux, test=ux)
        ay = Form(kernel=Kernel(one), trial=uy, test=uy)
        L = Form(kernel=Kernel(zero), test=u)
        assembler = Assembler([ax, ay, L], mesh)
        assembler.assemble()

        A = assembler.get_matrix()
        b = assembler.get_vector()

        # Dirichlet data u = x on left/right
        ue = Nodal(f=lambda X: X[:, 0], basis=u)

        return mesh, u, ue, A, b

    def _solve_once(self, u, A, b, mesh, ue, n_threads=None):
        system = LinearSystem(u, A=A, b=b)
        system.add_dirichlet_constraint('left', ue)
        system.add_dirichlet_constraint('right', ue)

        ctx = limit_math_threads(n_threads) if isinstance(n_threads, int) and n_threads > 0 else None
        t0 = time.perf_counter()
        if ctx is None:
            system.solve_system()
        else:
            with ctx:
                system.solve_system()
        dt = time.perf_counter() - t0

        sol = system.get_solution(as_function=True).data().ravel()
        return dt, sol

    def test_timing_compare_limit_math_threads(self):
        mesh, u, ue, A, b = self._build_poisson_2d(resolution=(32, 32))

        # Warm-up to initialize BLAS/OpenMP pools
        _ = self._solve_once(u, A, b, mesh, ue, n_threads=1)

        # Measure default threading
        dt_default, sol_default = self._solve_once(u, A, b, mesh, ue, n_threads=None)

        # Measure single-threaded
        dt_single, sol_single = self._solve_once(u, A, b, mesh, ue, n_threads=1)

        # Solutions must match regardless of threading
        self.assertTrue(np.allclose(sol_default, sol_single))

        # Emit timing info for inspection in test logs
        tp_before = current_threadpools()
        print("Threadpools before:", tp_before)
        print(f"Solve time default threads: {dt_default:.4f} s")
        print(f"Solve time 1 thread:        {dt_single:.4f} s")

        # Non-enforcing sanity check: make sure we actually timed something
        self.assertGreater(dt_default, 0)
        self.assertGreater(dt_single, 0)


if __name__ == '__main__':
    unittest.main()
