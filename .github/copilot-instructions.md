# quadmesh Copilot Instructions

**Purpose**: Adaptive 1D/2D quadtree FE toolkit for assembly, solving, plotting, and stochastic (GMRF) utilities.

## Architecture & Core Components

- **Core layout**: `src/mesh.py` (trees, `QuadCell`/`Interval`, hanging nodes, reference maps, flags), `src/fem.py` (`QuadFE`, `DofHandler`, `Basis`, shape/derivative helpers), `src/function.py` (`Map`, `Nodal`, `Explicit`, `Constant`, interpolation), `src/assembler.py` (`Kernel`, `Form`/`IIForm`/`IPForm`, `GaussRule`, shape eval, integration over cells/half-edges), `src/solver.py` (`LinearSystem` constraints for Dirichlet + hanging nodes), `src/spd.py` (SPD + Cholesky, CHOLMOD/modified LDLT), `src/gmrf.py` (cov kernels, KL/GMRF helpers), `src/plot.py` (mesh/function visualization).

## Workflow: Mesh → Element → DOF → Assembly → Solve

1. **Create mesh**: `mesh = QuadMesh(resolution=(m,n))` or `Mesh1D(resolution=(n,))` creates adaptive quadtree
2. **Define element**: `element = QuadFE(dim, etype)` where `etype ∈ {DQ0, Q1, Q2, Q3,...}` (DQ* = discontinuous)
3. **Build DOF handler**: `dh = DofHandler(mesh, element); dh.distribute_dofs(); dh.set_dof_vertices()`
4. **Create basis**: `u = Basis(dh, 'u'); v = Basis(dh, 'v')` carries trial/test identity via Basis naming (standard: u/ux/uy=trial, v/vx/vy=test)
5. **Assemble**: Define `Kernel([u, v], derivatives=[...], F=callable)` → `Form(trial=u, test=v, kernel=...)` → `Assembler([form], mesh)` → `assemble()`
6. **Solve**: `LinearSystem(u, A, b)` → `add_dirichlet_constraint(marker, func)` → `set_constraint_relation()` → `solve()`

## Key Implementation Details

- **Imports/pathing**: Code assumes `src` on `PYTHONPATH`; tests hardcode `/home/hans-werner/git/quadmesh/src/`. Run with: `PYTHONPATH=$(pwd)/src python script.py` or prepend `sys.path.append('/path/quadmesh/src')` (see `add_path.py`).
- **Elements & bases**: `QuadFE(dim, etype)` creates element; `DofHandler(mesh, element)` then `distribute_dofs()` + `set_dof_vertices()`; `Basis(dh, name)` holds trial/test identity via name convention; submeshes via `subforest_flag` parameter.
- **Meshes**: `QuadMesh`/`Mesh1D` are adaptive trees. Key ops: `cells.get_leaves()` (leaf cells), `cells.refine()` (subdivision), `cells.balance()` (enforce 1-level difference). Marking: `cell.mark(flag)` or `half_edge.mark(flag)` for region/refinement control.
- **Functions**: `Nodal(lambda_func, basis=...)` evaluates lambda at DOF nodes; `Explicit(func_obj, basis=...)` for explicit functions; `Constant(value)` for scalar constants; `Map(basis=dh, mesh=...)` for general FE functions; all support derivatives via interpolation.
- **Assembly pattern**: `Kernel([fields], derivatives=[...], F=callable)` defines integrand; `Form(trial=Basis, test=Basis, kernel=..., dmu={'dx'|'ds'}, flag=region_flag)` specifies form; `IIForm`/`IPForm` for interior/boundary penalties; bundle forms (list) into `Assembler(problem, mesh)` then call `assemble()`.
- **Derivatives**: Use tags like `'*x'`, `'*y'`, `'*xx'`, `'*xy'` in kernel definitions (parsed by `parse_derivative_info`); basis names use `'ux'`, `'uy'`, `'vx'`, `'vy'` etc. (name must match kernel derivative request or assertion fails).
- **Constraint workflow** (Dirichlet + hanging nodes): (1) Assemble A, b normally (full system). (2) Wrap in `LinearSystem(basis, A, b)`. (3) Call `add_dirichlet_constraint(bnd_marker, dirichlet_func, on_boundary=True)`. (4) Call `set_constraint_relation()`. (5) Call `constrain_matrix()` to apply both Dirichlet and hanging-node constraints. (6) `solve()` returns solution on *full* DOF set (constrained DOFs interpolated automatically). **Note**: `DofHandler.set_hanging_nodes()` is auto-invoked by `LinearSystem`; constrained DOFs stored in `constraints` dict keyed by DOF index.
- **SPD utilities**: For covariance/precision matrices use `SPDMatrix(C)` or `CholeskyDecomposition(A)`; dense path uses scipy.linalg, sparse path uses CHOLMOD (`sksparse.cholmod.cholesky`). If matrix not positive definite, code auto-falls back to modified LDLT (`modchol_ldlt`).
- **GMRF**: `gmrf.py` hosts covariance kernels (`gaussian`, `exponential`, `matern`, etc.), KL expansions, and assembly helpers; depends on assembler/basis stack. See `notes.txt` for aspirational features.
- **Plotting**: `Plot(mesh).mesh()`, `.contour(function)`, `.surface(function)` visualize; use `axis=ax` to embed. Expects mesh methods `bounding_box()`, `cells.get_leaves()`. Quickview opens figures immediately.

## Testing & Debugging

- **Run tests**: `PYTHONPATH=$(pwd)/src python -m unittest tests.test_assembler.test_assembler` (unittest, not pytest). Some tests stubbed (`pass`).
- **Common asserts**: Element dim must match mesh dim. Basis derivative names (u_x, v_y) must match kernel derivative requests. Quadrature order must be in allowed set (1D: {1,2,3,4,5,6}; quad: {1,4,9,16,25,36}; tri: {1,3,7,13}).

## Conventions & Pitfalls

- **Array shapes**: Routines expect numpy arrays (n, dim); `convert_to_array()` auto-converts tuples/Vertices—use it always.
- **Flags**: Drive region-specific assembly; must call `mark(flag)` on cells/half-edges *before* assembly.
- **DQ* elements**: Discontinuous (torn), no continuity across interfaces.
- **External deps**: numpy, scipy (sparse/linalg), matplotlib (plotting), scikit-sparse (CHOLMOD).
- **Examples**: `experiments/` for PDE/GMRF demos; `tests/test_assembler/*` for patterns; `add_path.py` shows path injection.
