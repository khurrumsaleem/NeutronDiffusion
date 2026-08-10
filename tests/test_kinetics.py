"""
Pytest suite for delayed neutron precursors and the implicit fission source.

The central check is an *exact* reduction: a one-group infinite medium (no
leakage, uniform flux) obeys the point kinetics equations term for term, with

    Lambda = 1 / (v * nu_sigf)      (neutron generation time)
    rho    = (k - 1) / k = 1 - Sigma_a / nu_sigf

so the transient computed by the diffusion solver can be compared against a
stiff ODE integration of point kinetics with no modelling error in between.
Any mistake in the effective-fission-spectrum algebra, the precursor update, or
the prompt/delayed split shows up directly.

Also covered:
  - a critical system with equilibrium precursors is a fixed point (the
    dt -> infinity limit of chi_eff);
  - delayed neutrons put a delayed-supercritical transient on a seconds
    timescale rather than the prompt ~1e-4 s one;
  - with no delayed data the implicit scheme reproduces the analytic prompt
    exponential;
  - all three solvers agree on the same infinite-medium transient, including
    on a graded unstructured mesh (which is what catches a cell-area weighting
    mistake in the FVM delayed source).
"""

import numpy as np
import pytest

import ndiffusion as nd

# ---------------------------------------------------------------------------
# Shared infinite-medium problem
# ---------------------------------------------------------------------------

BETA = np.array(nd.DELAYED_U235_6GROUP["Beta"])
LAM = np.array(nd.DELAYED_U235_6GROUP["Lambda"])
BETA_TOT = float(BETA.sum())

V = 2.2e5       # cm/s, ~thermal neutron speed
NUSIGF = 0.1570  # 1/cm
GEN_TIME = 1.0 / (V * NUSIGF)  # Lambda, ~2.9e-5 s


def infinite_medium(rho=0.0):
    """1-group material whose infinite-medium reactivity is *rho*.

    Removal is tuned to nu_sigf * (1 - rho), so k = 1/(1-rho) and
    rho = (k-1)/k exactly.  D is irrelevant - every boundary is reflective and
    the flux stays uniform, so there is no leakage.
    """
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 1
    m.D = [1.0]
    m.removal = [NUSIGF * (1.0 - rho)]
    m.scatter = [0.0]
    m.chi = [1.0]
    m.nusigf = [NUSIGF]
    m.velocity = [V]
    return m


def six_group_delayed():
    return nd.make_delayed_data(nd.DELAYED_U235_6GROUP, G=1, n_mat=1, chi=[1.0])


def point_kinetics(rho, t_end, n_eval=200):
    """Reference solution of the point kinetics equations.

    Starts from n = 1 with equilibrium precursors, so the t < 0 state is a
    genuine steady state and *rho* is a step insertion at t = 0.
    """
    solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp

    def rhs(_t, y):
        n, c = y[0], y[1:]
        dn = (rho - BETA_TOT) / GEN_TIME * n + np.sum(LAM * c)
        dc = BETA / GEN_TIME * n - LAM * c
        return np.concatenate([[dn], dc])

    c0 = BETA / (GEN_TIME * LAM)
    t_eval = np.linspace(0.0, t_end, n_eval)
    sol = solve_ivp(
        rhs, (0.0, t_end), np.concatenate([[1.0], c0]),
        t_eval=t_eval, method="Radau", rtol=1e-10, atol=1e-14,
    )
    return t_eval, sol.y[0]


def slab_solver(mats, delayed=None, cells=8, epsilon=1e-10, max_inner=500):
    """1-D infinite medium: reflective outer BC, symmetry at the inner edge."""
    edges = list(np.linspace(0.0, 10.0, cells + 1))
    kwargs = {} if delayed is None else {"delayed": delayed}
    return nd.TimeDependentSolver(
        mats=mats,
        medium_map=[0] * cells,
        edges_x=edges,
        geom=nd.Geometry.Slab,
        bc=[nd.BoundaryCondition(A=0.0, B=1.0)],
        initial_flux=[1.0] * cells,
        epsilon=epsilon,
        max_inner=max_inner,
        **kwargs,
    )


def amplitude(solver):
    """Flux amplitude - uniform across the medium, so any cell will do."""
    return solver.result().flux[0]


def run_to(solver, t_end, dt_fine=1e-4, t_fine=0.1, dt_coarse=1e-3):
    """Step to *t_end*, resolving the prompt jump with a finer dt.

    Exercises the variable-dt path: chi_eff depends on dt and is rebuilt
    whenever it changes.
    """
    n_fine = int(round(t_fine / dt_fine))
    n_coarse = int(round((t_end - t_fine) / dt_coarse))
    solver.run(dt_fine, n_fine)
    solver.run(dt_coarse, n_coarse)


# ---------------------------------------------------------------------------
# Point kinetics equivalence - the primary validation
# ---------------------------------------------------------------------------


class TestPointKinetics:
    """The infinite-medium diffusion transient IS point kinetics."""

    # Tolerances are set by backward Euler's first-order time error at the
    # coarse dt, not by the kinetics algebra: the faster the transient, the
    # more the O(dt) term costs.  $0.90 is close enough to prompt critical to
    # need the looser bound.
    @pytest.mark.parametrize("dollars,rel", [(0.5, 2e-3), (-0.5, 2e-3), (0.9, 5e-3)])
    def test_step_insertion_matches_point_kinetics(self, dollars, rel):
        rho = dollars * BETA_TOT

        solver = slab_solver(infinite_medium(0.0), six_group_delayed())
        solver.update_materials(infinite_medium(rho))
        run_to(solver, 1.0)

        _, n_ref = point_kinetics(rho, 1.0)
        assert solver.time == pytest.approx(1.0, rel=1e-9)
        assert amplitude(solver) == pytest.approx(n_ref[-1], rel=rel)

    def test_prompt_jump(self):
        """The fast rise right after a $0.5 step, against point kinetics.

        The textbook jump ratio beta/(beta-rho) = 2.0 is the idealised
        instantaneous limit; by 0.02 s the precursors have already started to
        matter, so the ODE reference is the honest comparison - and a tighter
        one.
        """
        rho = 0.5 * BETA_TOT
        t_end = 0.02

        solver = slab_solver(infinite_medium(0.0), six_group_delayed())
        solver.update_materials(infinite_medium(rho))
        solver.run(1e-5, int(round(t_end / 1e-5)))

        _, n_ref = point_kinetics(rho, t_end)
        assert amplitude(solver) == pytest.approx(n_ref[-1], rel=2e-3)
        # Still recognisably a prompt jump toward beta/(beta-rho) = 2.
        assert 1.8 < amplitude(solver) < 2.0

    def test_precursors_track_the_flux(self):
        """Precursors stay at equilibrium when nothing is perturbed."""
        delayed = six_group_delayed()
        solver = slab_solver(infinite_medium(0.0), delayed)

        c_equilibrium = BETA * NUSIGF / LAM  # C_i = beta_i * F / lambda_i
        assert np.allclose(solver.precursors[:6], c_equilibrium, rtol=1e-12)

        solver.run(1e-3, 100)
        assert np.allclose(solver.precursors[:6], c_equilibrium, rtol=1e-8)


class TestSteadyState:
    """A critical system with equilibrium precursors is an exact fixed point.

    This is the dt -> infinity limit of chi_eff, where the effective spectrum
    must collapse back to the total fission spectrum.  Large steps make it a
    sharp test: a sign or normalisation slip in the delayed algebra shows up
    immediately, while a small-dt transient test would mask it.
    """

    @pytest.mark.parametrize("dt", [1e-3, 1.0, 100.0])
    def test_flux_is_invariant(self, dt):
        solver = slab_solver(infinite_medium(0.0), six_group_delayed())
        solver.run(dt, 20)
        assert amplitude(solver) == pytest.approx(1.0, abs=1e-9)

    def test_prompt_only_critical_is_also_invariant(self):
        """Without delayed data the implicit fission source must still balance."""
        solver = slab_solver(infinite_medium(0.0))
        solver.run(1.0, 20)
        assert amplitude(solver) == pytest.approx(1.0, abs=1e-9)

    def test_scale_to_critical_makes_a_fixed_point(self):
        """A k-eigenvalue flux is only a steady state after scaling by 1/k."""
        cells = 40
        edges = list(np.linspace(0.0, 100.0, cells + 1))
        mats = infinite_medium(0.0)
        mats.removal = [0.1532]  # leaky sphere: k != 1
        bc = [nd.BoundaryCondition(A=1.0, B=0.0)]

        keig = nd.KEigenSolver(
            mats, [0] * cells, edges, nd.Geometry.Sphere, bc,
            epsilon=1e-10, max_outer=2000,
        )
        res = keig.solve()
        assert res.converged
        assert abs(res.keff - 1.0) > 1e-6  # otherwise the test proves nothing

        critical = nd.scale_to_critical(mats, res.keff)
        solver = nd.TimeDependentSolver(
            mats=critical, medium_map=[0] * cells, edges_x=edges,
            geom=nd.Geometry.Sphere, bc=bc, initial_flux=res.flux,
            epsilon=1e-12, max_inner=500, delayed=six_group_delayed(),
        )
        before = np.array(solver.result().flux)
        solver.run(0.1, 50)
        after = np.array(solver.result().flux)
        assert np.allclose(after, before, rtol=1e-6)

    def test_scale_to_critical_leaves_input_untouched(self):
        mats = infinite_medium(0.0)
        original = list(mats.nusigf)
        scaled = nd.scale_to_critical(mats, 1.25)
        assert mats.nusigf == original
        assert scaled.nusigf == pytest.approx([v / 1.25 for v in original])


class TestTimescaleSeparation:
    """Delayed neutrons are what put the transient on a physical timescale."""

    def test_delayed_transient_is_slow(self):
        """A $0.5 insertion must take seconds, not prompt lifetimes."""
        rho = 0.5 * BETA_TOT
        solver = slab_solver(infinite_medium(0.0), six_group_delayed())
        solver.update_materials(infinite_medium(rho))
        solver.run(1e-4, 1000)  # 0.1 s

        # Past the prompt jump (~2x) but nowhere near a prompt excursion.
        assert 1.5 < amplitude(solver) < 5.0

    def test_prompt_only_transient_is_fast(self):
        """The same insertion without precursors runs away on ~Lambda/rho."""
        rho = 0.5 * BETA_TOT
        solver = slab_solver(infinite_medium(0.0))
        solver.update_materials(infinite_medium(rho))
        solver.run(1e-6, 100)  # only 1e-4 s

        # Prompt period is Lambda/rho ~ 8.9e-3 s, so 1e-4 s already grows
        # measurably - and by 0.1 s the prompt-only answer is astronomically
        # larger than the delayed one above.
        assert amplitude(solver) > 1.01

    def test_prompt_only_matches_analytic_exponential(self):
        """Prompt-only infinite medium: phi(t) = exp(rho / Lambda * t)."""
        rho = 0.2 * BETA_TOT
        solver = slab_solver(infinite_medium(rho))
        dt, n = 1e-7, 1000
        solver.run(dt, n)

        t = dt * n
        expected = np.exp(rho / GEN_TIME * t)
        assert amplitude(solver) == pytest.approx(expected, rel=1e-3)

    def test_subcritical_decays(self):
        rho = -1.0 * BETA_TOT
        solver = slab_solver(infinite_medium(0.0), six_group_delayed())
        solver.update_materials(infinite_medium(rho))
        run_to(solver, 1.0)
        assert amplitude(solver) < 0.6


# ---------------------------------------------------------------------------
# Cross-solver agreement
# ---------------------------------------------------------------------------


def quad_mesh(x_edges, y_edges, bc_tag=0):
    """Rectilinear quad mesh from explicit edge coordinates.

    Taking edges rather than a cell count lets a test grade the mesh, which is
    what exposes an incorrect cell-area weighting in the FVM delayed source:
    on a uniform mesh the area cancels out of the comparison.
    """
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    vx, vy = [], []
    for x in x_edges:
        for y in y_edges:
            vx.append(float(x))
            vy.append(float(y))

    def vid(i, j):
        return i * (ny + 1) + j

    cell_vertices, cell_offsets, mat_ids = [], [0], []
    for i in range(nx):
        for j in range(ny):
            cell_vertices += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            cell_offsets.append(len(cell_vertices))
            mat_ids.append(0)

    bface_v0, bface_v1, bface_bc_tag = [], [], []
    for i in range(nx):
        bface_v0.append(vid(i, 0)); bface_v1.append(vid(i + 1, 0))
        bface_bc_tag.append(bc_tag)
    for j in range(ny):
        bface_v0.append(vid(nx, j)); bface_v1.append(vid(nx, j + 1))
        bface_bc_tag.append(bc_tag)
    for i in range(nx - 1, -1, -1):
        bface_v0.append(vid(i + 1, ny)); bface_v1.append(vid(i, ny))
        bface_bc_tag.append(bc_tag)
    for j in range(ny - 1, -1, -1):
        bface_v0.append(vid(0, j + 1)); bface_v1.append(vid(0, j))
        bface_bc_tag.append(bc_tag)

    mesh = nd.UnstructuredMesh2D()
    mesh.vx = vx
    mesh.vy = vy
    mesh.cell_vertices = cell_vertices
    mesh.cell_offsets = cell_offsets
    mesh.material_id = mat_ids
    mesh.bface_v0 = bface_v0
    mesh.bface_v1 = bface_v1
    mesh.bface_bc_tag = bface_bc_tag
    return mesh


class TestSolverAgreement:
    """All three time-dependent solvers on the same infinite-medium transient.

    This is the worst case for the implicit fission iteration: an exactly
    critical medium with no leakage, at a dt where the ``1/(v*dt)`` diagonal
    term is only ~3% of Sigma_a and so contributes almost no diagonal
    dominance.  Unaccelerated, the fission fixed point converges at roughly k
    per sweep and this needed ``max_inner`` in the thousands; the Aitken
    extrapolation in ``FissionAccelerator`` brings it down to the value below.
    See ``TestAcceleration`` for the direct check.
    """

    RHO = 0.5 * BETA_TOT
    DT = 1e-3
    N_STEPS = 200  # 0.2 s
    # Tight, since every solver integrates the identical ODE - the residual
    # difference is the inner-iteration tolerance, not discretization.
    EPS = 1e-12
    REL = 1e-8
    MAX_INNER = 200

    def reference(self):
        solver = slab_solver(
            infinite_medium(0.0), six_group_delayed(), epsilon=self.EPS,
            max_inner=self.MAX_INNER,
        )
        solver.update_materials(infinite_medium(self.RHO))
        solver.run(self.DT, self.N_STEPS)
        return amplitude(solver)

    def test_structured_2d_matches_1d(self):
        nx = ny = 3
        edges_x = list(np.linspace(0.0, 6.0, nx + 1))
        edges_y = list(np.linspace(0.0, 6.0, ny + 1))
        reflective = [nd.BoundaryCondition(A=0.0, B=1.0)]

        solver = nd.TimeDependentSolver2D(
            mats=infinite_medium(0.0), medium_map=[0] * (nx * ny),
            edges_x=edges_x, edges_y=edges_y, geom=nd.Geometry2D.XY,
            bc_x=reflective, bc_y=reflective,
            initial_flux=[1.0] * (nx * ny),
            epsilon=self.EPS, max_inner=self.MAX_INNER,
            delayed=six_group_delayed(),
        )
        solver.update_materials(infinite_medium(self.RHO))
        solver.run(self.DT, self.N_STEPS)

        assert solver.result().flux[0] == pytest.approx(
            self.reference(), rel=self.REL
        )

    def test_unstructured_on_a_graded_mesh_matches_1d(self):
        # Deliberately non-uniform: cell areas span a factor of 9, so any
        # missing or spurious area weight in the delayed source is visible.
        x_edges = [0.0, 1.0, 3.0, 6.0]
        y_edges = [0.0, 1.0, 3.0, 6.0]
        mesh = quad_mesh(x_edges, y_edges)
        n_cells = (len(x_edges) - 1) * (len(y_edges) - 1)
        reflective = [nd.BoundaryCondition(A=0.0, B=1.0)]

        solver = nd.TimeDependentSolverUnstructured2D(
            mats=infinite_medium(0.0), mesh=mesh, bc=reflective,
            initial_flux=[1.0] * n_cells,
            epsilon=self.EPS, max_inner=self.MAX_INNER,
            delayed=six_group_delayed(),
        )
        solver.update_materials(infinite_medium(self.RHO))
        solver.run(self.DT, self.N_STEPS)

        flux = np.array(solver.result().flux)
        # Flux must stay uniform despite the graded mesh.  A missing or
        # spurious cell-area factor in the delayed source would scale each
        # cell differently, giving an O(1) spread rather than this one, which
        # tracks the point-Gauss-Seidel tolerance exactly.
        assert np.allclose(flux, flux[0], rtol=1e-9)
        assert flux[0] == pytest.approx(self.reference(), rel=self.REL)

    def test_unstructured_precursors_are_per_unit_volume(self):
        """Precursors must not pick up a cell-area factor on a graded mesh."""
        x_edges = [0.0, 1.0, 4.0]
        y_edges = [0.0, 1.0, 4.0]
        mesh = quad_mesh(x_edges, y_edges)
        n_cells = 4
        solver = nd.TimeDependentSolverUnstructured2D(
            mats=infinite_medium(0.0), mesh=mesh,
            bc=[nd.BoundaryCondition(A=0.0, B=1.0)],
            initial_flux=[1.0] * n_cells,
            epsilon=1e-10, max_inner=500, delayed=six_group_delayed(),
        )
        # Uniform flux -> identical precursors in every cell, whatever its area.
        c = np.array(solver.precursors).reshape(n_cells, 6)
        assert np.allclose(c, c[0], rtol=1e-12)
        assert np.allclose(c[0], BETA * NUSIGF / LAM, rtol=1e-12)


class TestAcceleration:
    """The Aitken extrapolation on the implicit-fission fixed point.

    Without it, the worst case below (exactly critical, zero leakage, dt where
    1/(v*dt) is ~3% of Sigma_a) converges at roughly k per sweep and needs
    thousands of inner iterations per step.  These tests pin both halves of the
    claim: it converges within a small budget, and it converges to the *same*
    answer a very large budget would give.
    """

    RHO = 0.5 * BETA_TOT

    def worst_case(self, max_inner):
        solver = slab_solver(
            infinite_medium(0.0), six_group_delayed(),
            epsilon=1e-12, max_inner=max_inner,
        )
        solver.update_materials(infinite_medium(self.RHO))
        solver.run(1e-3, 200)
        return amplitude(solver)

    def test_converges_within_a_small_budget(self, capfd):
        """No step may hit max_inner - the solver warns on stderr when one does."""
        amp = self.worst_case(max_inner=50)
        err = capfd.readouterr().err
        assert "did not converge" not in err, err
        assert amp == pytest.approx(self.worst_case(max_inner=5000), rel=1e-9)

    def test_extrapolation_does_not_shift_the_answer(self):
        """Acceleration changes the path to the fixed point, not the fixed point."""
        assert self.worst_case(50) == pytest.approx(self.worst_case(2000), rel=1e-9)

    def test_prompt_only_still_exact(self):
        """Extrapolating must not disturb an already-fast iteration."""
        rho = 0.2 * BETA_TOT
        solver = slab_solver(infinite_medium(rho), max_inner=50)
        dt, n = 1e-7, 1000
        solver.run(dt, n)
        expected = np.exp(rho / GEN_TIME * (dt * n))
        assert amplitude(solver) == pytest.approx(expected, rel=1e-3)

    def test_steady_state_still_exact(self):
        """A fixed point has zero iterate change - nothing to extrapolate."""
        solver = slab_solver(
            infinite_medium(0.0), six_group_delayed(), max_inner=50
        )
        solver.run(100.0, 20)
        assert amplitude(solver) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Fission-matrix mode
# ---------------------------------------------------------------------------


# Thermal removal that makes the separable 2-group medium below critical
# (k_inf = 0.999985 with chi = (0.7, 0.3), nuSigf = (0.005, 0.14)).  A
# supercritical medium would simply overflow over the transients here.
TWO_GROUP_REMOVAL_2 = 0.1697


def two_group_slab(chi, nusigf, rho_scale=1.0):
    """2-group infinite medium, standard or fission-matrix representation.

    *rho_scale* scales the thermal removal: below 1 adds reactivity.
    """
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 2
    m.D = [1.0, 1.0]
    # Fast group leaves only by down-scatter; thermal group absorbs.
    m.removal = [0.02, TWO_GROUP_REMOVAL_2 * rho_scale]
    m.scatter = [0.0, 0.0, 0.02, 0.0]  # scatter[g_to][g_from]: 1 -> 2 only
    m.chi = list(chi)
    m.nusigf = list(nusigf)
    m.velocity = [1.0e7, 2.2e5]
    return m


class TestFissionMatrixMode:
    """Delayed neutrons with `nusigf` holding a full fission transfer matrix.

    There is no separable chi to split, so the prompt/delayed decomposition is
    applied to the matrix itself: the delayed yield `beta_i * chi_d,i[g_to] *
    P[g_from]` is subtracted from the tabulated (total) matrix and the portion
    emitted within the step added back, where `P[g_from]` is the column sum -
    the total neutrons emitted per fission caused by a group-`g_from` neutron.

    The decisive test is that a *separable* matrix reproduces the standard-mode
    transient exactly: `F[g][g'] = chi_g nuSigf_g'` and `sum_g chi_g = 1` make
    the column sum equal `nuSigf_g'`, so both branches must agree to round-off.
    """

    CHI = [0.7, 0.3]
    NUSIGF = [0.005, 0.14]
    DT, N_STEPS = 1e-3, 100

    def delayed(self, chi_d=None):
        spec = dict(nd.DELAYED_U235_6GROUP)
        spec["ChiDelayed"] = self.CHI if chi_d is None else chi_d
        return nd.make_delayed_data(spec, G=2, n_mat=1)

    def standard(self):
        return two_group_slab(self.CHI, self.NUSIGF)

    def matrix(self):
        """The same physics with F[g_to][g_from] = chi_g_to * nuSigf_g_from."""
        f = [self.CHI[gt] * self.NUSIGF[gf] for gt in range(2) for gf in range(2)]
        mats = two_group_slab([0.0, 0.0], f)
        assert sum(mats.chi) == 0.0  # fission-matrix mode is keyed on this
        return mats

    def build(self, mats, delayed):
        cells = 6
        edges = list(np.linspace(0.0, 10.0, cells + 1))
        return nd.TimeDependentSolver(
            mats=mats, medium_map=[0] * cells, edges_x=edges,
            geom=nd.Geometry.Slab,
            bc=[nd.BoundaryCondition(A=0.0, B=1.0)] * 2,
            initial_flux=[1.0, 1.0] * cells,
            epsilon=1e-12, max_inner=500, delayed=delayed,
        )

    def run(self, mats, delayed, perturb=None):
        solver = self.build(mats, delayed)
        if perturb is not None:
            solver.update_materials(perturb)
        solver.run(self.DT, self.N_STEPS)
        return np.array(solver.result().flux), np.array(solver.precursors)

    def test_separable_matrix_matches_standard_mode(self):
        flux_s, prec_s = self.run(self.standard(), self.delayed())
        flux_m, prec_m = self.run(self.matrix(), self.delayed())
        assert np.allclose(flux_m, flux_s, rtol=1e-10)
        assert np.allclose(prec_m, prec_s, rtol=1e-10)

    def test_separable_matrix_matches_under_a_perturbation(self):
        """Agreement must survive update_materials, not just the initial state."""
        pert_s = two_group_slab(self.CHI, self.NUSIGF, rho_scale=0.999)
        f = [self.CHI[gt] * self.NUSIGF[gf] for gt in range(2) for gf in range(2)]
        pert_m = two_group_slab([0.0, 0.0], f, rho_scale=0.999)

        flux_s, _ = self.run(self.standard(), self.delayed(), perturb=pert_s)
        flux_m, _ = self.run(self.matrix(), self.delayed(), perturb=pert_m)
        assert np.allclose(flux_m, flux_s, rtol=1e-10)
        # The perturbation must actually do something, or this proves nothing.
        assert not np.allclose(flux_s, self.run(self.standard(), self.delayed())[0])

    def test_equilibrium_precursors_use_the_column_sum(self):
        """Production in matrix mode is sum_g F[g][g'], not the first row.

        Read at t = 0: the initial flux is 1 in both groups, so the production
        rate is exactly the sum of the column sums.
        """
        solver = self.build(self.matrix(), self.delayed())
        expected = BETA * sum(self.NUSIGF) / LAM
        assert np.allclose(solver.precursors[:6], expected, rtol=1e-10)

    def test_non_separable_matrix_runs(self):
        """A genuinely non-separable matrix - no chi vector reproduces it."""
        # Fast fission emits harder than thermal fission: the two columns have
        # different spectra, so F is not an outer product of chi and nuSigf.
        f = [0.9 * 0.005, 0.6 * 0.14,
             0.1 * 0.005, 0.4 * 0.14]
        mats = two_group_slab([0.0, 0.0], f)

        # Column sums are unchanged, so production matches the separable case
        # even though the emission spectra differ per causing group.
        solver = self.build(mats, self.delayed())
        assert np.allclose(
            solver.precursors[:6], BETA * sum(self.NUSIGF) / LAM, rtol=1e-10
        )

        flux, _ = self.run(mats, self.delayed())
        assert np.all(np.isfinite(flux)) and np.all(flux > 0.0)

        # The initial flux (1, 1) is not the eigenvector, so the group shape
        # relaxes within microseconds and then the amplitude decays slowly.
        # The relaxed shape must be the k-eigenvector of the same fission
        # matrix - which is what pins down matrix-mode fission in the transient
        # path, not just in the precursor source.
        cells = 6
        edges = list(np.linspace(0.0, 10.0, cells + 1))
        keig = nd.KEigenSolver(
            mats, [0] * cells, edges, nd.Geometry.Slab,
            [nd.BoundaryCondition(A=0.0, B=1.0)] * 2,
            epsilon=1e-12, max_outer=5000,
        )
        res = keig.solve()
        assert res.converged
        assert res.keff < 1.0  # subcritical, so the amplitude decays

        # Not exact: the delayed source carries chi_d rather than the matrix's
        # own emission spectrum, so the dynamic shape sits slightly off the
        # static eigenvector.  A wrong matrix would be off by far more than 1%.
        ratio_transient = flux[0] / flux[1]
        ratio_eigen = res.flux[0] / res.flux[1]
        assert ratio_transient == pytest.approx(ratio_eigen, rel=1e-2)

    def test_delayed_yield_must_fit_inside_the_matrix(self):
        """nusigf is the total matrix, so the delayed part must fit in it."""
        # Delayed neutrons all in group 0, but the matrix emits almost nothing
        # there - beta * chi_d * P would exceed F[0][g'].
        f = [1e-6, 1e-6, 0.005, 0.14]
        mats = two_group_slab([0.0, 0.0], f)
        with pytest.raises(ValueError, match="exceeds the tabulated fission"):
            self.run(mats, self.delayed(chi_d=[1.0, 0.0]))

    def test_chi_delayed_must_be_normalised(self):
        """Catches defaulting chi_delayed from the all-zero matrix-mode chi."""
        delayed = self.delayed()
        delayed.chi_delayed = [0.0] * (6 * 2)
        with pytest.raises(ValueError, match="must sum to 1"):
            self.run(self.matrix(), delayed)


class TestPromptSpectrumDerivation:
    """`chi_prompt` defaults so that prompt + delayed sum back to `chi`.

    Without that, supplying a delayed spectrum different from `Materials.chi`
    would silently move the steady state: `chi_eff` at large dt would no longer
    equal the total spectrum, and a critical system would stop being a fixed
    point.
    """

    def solver_for(self, chi_d):
        spec = dict(nd.DELAYED_U235_6GROUP)
        spec["ChiDelayed"] = chi_d
        delayed = nd.make_delayed_data(spec, G=2, n_mat=1)
        mats = two_group_slab([0.7, 0.3], [0.005, 0.14])
        cells = 6
        edges = list(np.linspace(0.0, 10.0, cells + 1))
        keig = nd.KEigenSolver(
            mats, [0] * cells, edges, nd.Geometry.Slab,
            [nd.BoundaryCondition(A=0.0, B=1.0)] * 2,
            epsilon=1e-12, max_outer=2000,
        )
        res = keig.solve()
        assert res.converged
        return nd.TimeDependentSolver(
            mats=nd.scale_to_critical(mats, res.keff), medium_map=[0] * cells,
            edges_x=edges, geom=nd.Geometry.Slab,
            bc=[nd.BoundaryCondition(A=0.0, B=1.0)] * 2,
            initial_flux=res.flux, epsilon=1e-12, max_inner=500, delayed=delayed,
        )

    @pytest.mark.parametrize("chi_d", [[0.7, 0.3], [0.4, 0.6], [1.0, 0.0]])
    def test_steady_state_holds_for_any_delayed_spectrum(self, chi_d):
        solver = self.solver_for(chi_d)
        before = np.array(solver.result().flux)
        solver.run(10.0, 20)  # large dt: the chi_eff -> total-spectrum limit
        assert np.allclose(solver.result().flux, before, rtol=1e-6)

    def test_delayed_spectrum_cannot_exceed_the_total(self):
        """beta_i * chi_d,i must fit inside chi, or the prompt part goes negative."""
        spec = {"Beta": [0.5], "Lambda": [0.08], "ChiDelayed": [0.0, 1.0]}
        delayed = nd.make_delayed_data(spec, G=2, n_mat=1)
        mats = two_group_slab([0.9, 0.1], [0.005, 0.14])  # only 0.1 in group 2
        cells = 4
        edges = list(np.linspace(0.0, 10.0, cells + 1))
        with pytest.raises(ValueError, match="derived prompt spectrum"):
            nd.TimeDependentSolver(
                mats=mats, medium_map=[0] * cells, edges_x=edges,
                geom=nd.Geometry.Slab,
                bc=[nd.BoundaryCondition(A=0.0, B=1.0)] * 2, delayed=delayed,
            )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


class TestDelayedDataValidation:
    def test_beta_must_sum_below_one(self):
        delayed = six_group_delayed()
        delayed.beta = [0.2] * 6  # sums to 1.2
        with pytest.raises(ValueError, match="delayed fraction"):
            slab_solver(infinite_medium(0.0), delayed)

    def test_lambda_must_be_positive(self):
        delayed = six_group_delayed()
        lam = list(delayed.lambda_)
        lam[0] = 0.0
        delayed.lambda_ = lam
        with pytest.raises(ValueError, match="lambda"):
            slab_solver(infinite_medium(0.0), delayed)

    def test_wrong_beta_size_is_rejected(self):
        delayed = six_group_delayed()
        delayed.beta = [1e-3, 1e-3]  # n_precursor is 6
        with pytest.raises(ValueError, match="beta"):
            slab_solver(infinite_medium(0.0), delayed)

    def test_wrong_initial_precursor_length(self):
        edges = list(np.linspace(0.0, 10.0, 9))
        with pytest.raises(ValueError, match="initial_precursors"):
            nd.TimeDependentSolver(
                mats=infinite_medium(0.0), medium_map=[0] * 8, edges_x=edges,
                geom=nd.Geometry.Slab, bc=[nd.BoundaryCondition(A=0.0, B=1.0)],
                delayed=six_group_delayed(), initial_precursors=[0.0] * 5,
            )

    def test_update_materials_rejects_a_shape_change(self):
        solver = slab_solver(infinite_medium(0.0), six_group_delayed())
        two_group = infinite_medium(0.0)
        two_group.n_groups = 2
        two_group.D = [1.0, 1.0]
        two_group.removal = [0.1, 0.1]
        two_group.scatter = [0.0] * 4
        two_group.chi = [1.0, 0.0]
        two_group.nusigf = [0.1, 0.1]
        two_group.velocity = [V, V]
        with pytest.raises(ValueError, match="n_mat or n_groups"):
            solver.update_materials(two_group)

    def test_explicit_initial_precursors_are_used(self):
        edges = list(np.linspace(0.0, 10.0, 9))
        seeded = [0.0] * (8 * 6)
        solver = nd.TimeDependentSolver(
            mats=infinite_medium(0.0), medium_map=[0] * 8, edges_x=edges,
            geom=nd.Geometry.Slab, bc=[nd.BoundaryCondition(A=0.0, B=1.0)],
            initial_flux=[1.0] * 8,
            delayed=six_group_delayed(), initial_precursors=seeded,
        )
        assert np.allclose(solver.precursors, 0.0)
        # Zero precursors in a critical system means the delayed source is
        # missing, so the flux drops rather than holding steady.
        solver.run(1e-3, 100)
        assert solver.result().flux[0] < 0.999


class TestMakeDelayedData:
    def test_broadcast_to_multiple_materials(self):
        delayed = nd.make_delayed_data(
            nd.DELAYED_U235_6GROUP, G=2, n_mat=3, chi=[1.0, 0.0]
        )
        assert delayed.n_precursor == 6
        assert len(delayed.beta) == 3 * 6
        assert len(delayed.chi_delayed) == 3 * 6 * 2
        assert delayed.chi_prompt == []  # falls back to Materials.chi

    def test_per_material_specs(self):
        fuel = dict(nd.DELAYED_U235_6GROUP)
        inert = {"Beta": [0.0] * 6, "Lambda": nd.DELAYED_U235_6GROUP["Lambda"]}
        delayed = nd.make_delayed_data([fuel, inert], G=1, chi=[1.0])
        assert delayed.beta[:6] == pytest.approx(BETA)
        assert delayed.beta[6:] == pytest.approx([0.0] * 6)

    def test_explicit_chi_prompt_is_kept(self):
        spec = dict(nd.DELAYED_U235_6GROUP)
        spec["ChiPrompt"] = [0.7, 0.3]
        spec["ChiDelayed"] = [0.9, 0.1]
        delayed = nd.make_delayed_data(spec, G=2, n_mat=1)
        assert delayed.chi_prompt == pytest.approx([0.7, 0.3])
        assert delayed.chi_delayed[:2] == pytest.approx([0.9, 0.1])

    def test_mismatched_lambda_between_materials_raises(self):
        a = dict(nd.DELAYED_U235_6GROUP)
        b = {"Beta": BETA.tolist(), "Lambda": (LAM * 2).tolist()}
        with pytest.raises(ValueError, match="Lambda differs"):
            nd.make_delayed_data([a, b], G=1, chi=[1.0])

    def test_mismatched_precursor_count_raises(self):
        a = dict(nd.DELAYED_U235_6GROUP)
        b = {"Beta": [1e-3], "Lambda": [0.1]}
        with pytest.raises(ValueError, match="precursor count"):
            nd.make_delayed_data([a, b], G=1, chi=[1.0])

    def test_missing_chi_fallback_raises(self):
        with pytest.raises(ValueError, match="ChiDelayed"):
            nd.make_delayed_data(nd.DELAYED_U235_6GROUP, G=1, n_mat=1)

    def test_beta_lambda_length_mismatch_raises(self):
        spec = {"Beta": [1e-3, 2e-3], "Lambda": [0.1]}
        with pytest.raises(ValueError, match="length n_precursor"):
            nd.make_delayed_data(spec, G=1, n_mat=1, chi=[1.0])
