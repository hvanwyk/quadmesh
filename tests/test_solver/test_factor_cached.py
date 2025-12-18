import unittest
import numpy as np

from assembler import Form, Kernel, Assembler
from mesh import Mesh1D
from fem import DofHandler, Basis, QuadFE
from function import Constant, Nodal
from solver import LinearSystem


class TestCachedFactor(unittest.TestCase):
    def test_multi_rhs_reuses_factor(self):
        """Solve two RHS with one factorization and verify scaling."""
        mesh = Mesh1D(resolution=(8,))
        element = QuadFE(1, 'Q1')
        dh = DofHandler(mesh, element)
        dh.distribute_dofs()

        u = Basis(dh, 'u')
        ux = Basis(dh, 'ux')

        one = Constant(1)
        a = Form(kernel=Kernel(one), trial=ux, test=ux)
        f = Constant(1)
        L = Form(kernel=Kernel(f), test=u)

        assembler = Assembler([a, L], mesh)
        assembler.assemble()

        A = assembler.get_matrix()
        b = assembler.get_vector()

        # Dirichlet boundary conditions u(0)=u(1)=0
        mesh.mark_region('left', lambda x: np.abs(x) < 1e-9, on_boundary=True)
        mesh.mark_region('right', lambda x: np.abs(1 - x) < 1e-9, on_boundary=True)

        system = LinearSystem(u, A=A, b=None)
        system.add_dirichlet_constraint('left', 0)
        system.add_dirichlet_constraint('right', 0)

        b_stack = np.column_stack([b, 2.0 * b])

        system.solve_system(b_stack, factor=True)
        sol = system.get_solution(as_function=True).data()

        ue = Nodal(f=lambda x: 0.5 * (x - x**2), basis=u).data()

        self.assertTrue(np.allclose(sol[:, 0], ue.ravel()))
        self.assertTrue(np.allclose(sol[:, 1], 2.0 * ue.ravel()))


if __name__ == '__main__':
    unittest.main()
