"""Independent full-site matrix oracles for bounded synthetic CPU lattices."""

import itertools
import unittest
import weakref
from unittest import mock

import numpy as np

from lqcdweave.contracts.elemental import GaugePath
from lqcdweave.production.elemental import ElementalGenerator, generate_baryon_elemental
from lqcdweave.production.elemental import re_combine
from lqcdweave.production.gauge import project_su3, stout_smear
from lqcdweave.sampling.mode_weights import dilution_pair_weights
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


class Handle:
    def __init__(self, array):
        self.array = array

    def load(self, key):
        return self.array


class _EstimatedSizeArray(np.ndarray):
    """Test-only array view with a synthetic size hint for batch estimation."""

    def __new__(cls, source, estimated_size):
        result = np.asarray(source).view(cls)
        result._estimated_size = estimated_size
        return result

    def __array_finalize__(self, source):
        self._estimated_size = getattr(source, "_estimated_size", None)

    @property
    def size(self):
        return self._estimated_size


def fixture():
    rng = np.random.default_rng(491)
    sizes = (3, 2, 2, 3)
    lx, ly, lz, lt = sizes
    raw = rng.normal(size=(lt, lz, ly, lx, 4, 3, 3)) + 1j*rng.normal(size=(lt, lz, ly, lx, 4, 3, 3))
    gauge = np.linalg.qr(raw)[0]
    gauge[..., :, 0] /= np.linalg.det(gauge)[..., None]
    vectors = rng.normal(size=(lt, 3, lz, ly, lx, 3)) + 1j*rng.normal(size=(lt, 3, lz, ly, lx, 3))
    points = np.array([[(t % lx, 0, 0), ((t+1) % lx, 0, 0), (0, 1, 1)] for t in range(lt)])
    return sizes, gauge, vectors, points


def reference_matrix(gauge, steps, t, momentum=(0, 0, 0), phase="midpoint", boundary="periodic"):
    """Explicit site/link loops; does not call transport/phase/recipe helpers."""
    lt, lz, ly, lx = gauge.shape[:4]
    volume = lx*ly*lz
    result = np.zeros((3*volume, 3*volume), complex)
    axes, signs = (0,1,2,0,1,2,3,3), (1,1,1,-1,-1,-1,1,-1)
    delta = np.zeros(4, int)
    for s in steps:
        delta[axes[s]] += signs[s]
    for z, y, x in np.ndindex(lz, ly, lx):
        position = np.array([x,y,z,t])
        matrix = np.eye(3, dtype=complex)
        valid = True
        for s in steps:
            d, sign = axes[s], signs[s]
            if sign < 0:
                position[d] -= 1
            if boundary == "open" and not 0 <= position[3] < lt:
                valid = False
                break
            px, py, pz, pt = position % np.array([lx,ly,lz,lt])
            link = gauge[pt,pz,py,px,d]
            matrix = matrix @ (link if sign > 0 else link.conj().T)
            if sign > 0:
                position[d] += 1
            if boundary == "open" and not 0 <= position[3] < lt:
                valid = False
                break
        if not valid:
            continue
        px, py, pz = position[:3] % np.array([lx,ly,lz])
        i, j = (z*ly*lx+y*lx+x)*3, (pz*ly*lx+py*lx+px)*3
        coordinate = np.array([x,y,z], float) + (delta[:3]/2 if phase == "midpoint" else 0)
        angle = 0 if phase == "none" else 2*np.pi*np.sum(np.asarray(momentum)*coordinate/np.array([lx,ly,lz]))
        result[i:i+3,j:j+3] = np.exp(1j*angle)*matrix
    return result


def point_basis(points, sizes):
    lx, ly, lz = sizes[:3]
    out = np.zeros((lx*ly*lz*3, len(points)*3), complex)
    for i, (x,y,z) in enumerate(points):
        j = (z*ly*lx+y*lx+x)*3
        out[j:j+3, i*3:i*3+3] = np.eye(3)
    return out


class ElementalTests(unittest.TestCase):
    def setUp(self):
        self.sizes, self.u, self.v, self.p = fixture()

    def gen(self, paths, **kwargs):
        return ElementalGenerator(self.sizes, Handle(self.u), Handle(self.v), point=Handle(self.p),
                                  paths=paths, path_version="v2", ne_block_size=2, **kwargs).load("cfg")

    def test_all_sectors_against_full_site_matrices(self):
        paths = [(), (0,), (3,), (0,1,3,4), (6,0), (7,), (0,3), (6,7)]
        moms = [(0,0,0), (1,-1,1)]
        for phase in ("midpoint", "anchor", "none"):
            gen = self.gen(paths, momentum_list=moms, phase_convention=phase)
            for t in range(self.sizes[3]):
                out = gen.calc_path_blocks(t, ("v2v","p2v","v2p","p2p"))
                for pi, steps in enumerate(paths):
                    end = (t + sum(1 if s == 6 else -1 if s == 7 else 0 for s in steps)) % self.sizes[3]
                    bases_l = {"v": self.v[t].reshape(3,-1).T, "p": point_basis(self.p[t], self.sizes)}
                    bases_r = {"v": self.v[end].reshape(3,-1).T, "p": point_basis(self.p[end], self.sizes)}
                    for mi, mom in enumerate(moms):
                        matrix = reference_matrix(self.u, steps, t, mom, phase)
                        for sector in out:
                            expected = bases_l[sector[0]].conj().T @ matrix @ bases_r[sector[-1]]
                            actual = out[sector][pi][mi]
                            if sector == "p2p":
                                actual = actual.to_dense()
                            np.testing.assert_allclose(actual.reshape(expected.shape), expected, atol=8e-14, rtol=2e-13)

    def test_open_boundary_checks_excursion_and_closed_loop(self):
        gen = self.gen([(7,6), (6,7), (0,1,3,4)], path_boundary="open")
        out = gen.calc_path_blocks(0, ("v2v","p2v","v2p","p2p"))
        for sector in out:
            first = out[sector][0][0]
            first = first.to_dense() if sector == "p2p" else first
            np.testing.assert_array_equal(first, 0)
        pp = out["p2p"][2][0].to_dense().reshape(9,9)
        self.assertGreater(np.linalg.norm(pp - np.eye(9)), .1)

    def test_streaming_matches_buffers_and_borrowing(self):
        gen = self.gen([(), (0,), (6,)], momentum_list=[(1,0,0)])
        full = gen.calc_path_blocks(0, ("v2v","v2p","p2v","p2p"))
        target = {k: np.zeros_like(v) for k,v in full.items() if k != "p2p"}
        seen = []
        def write(pi, mi, sector, slices, value):
            if sector == "p2p":
                np.testing.assert_allclose(value.to_dense(), full[sector][pi][mi].to_dense())
            else:
                target[sector][(pi,mi)+slices] = value
            seen.append(sector)
        self.assertIsNone(gen.calc_path_blocks(0, tuple(full), write=write))
        for k, v in target.items():
            np.testing.assert_allclose(v, full[k])
        borrowed = full["v2v"]
        owned = borrowed.copy()
        self.assertIs(gen.calc_path_blocks(1)["v2v"], borrowed)
        self.assertGreater(np.linalg.norm(borrowed-owned), .1)
        self.assertFalse(hasattr(gen, "iter_path_blocks"))

    def test_point_only_does_not_require_vectors(self):
        gen = ElementalGenerator(self.sizes, Handle(self.u), point=Handle(self.p),
                                 paths=[()], path_version="v2").load("cfg")
        np.testing.assert_allclose(gen.calc_path_blocks(0, ("p2p",))["p2p"][0][0].to_dense().reshape(9,9), np.eye(9))

    def test_ordered_derivative_reference(self):
        moms = [(0,0,0), (1,1,0)]
        deriv = ElementalGenerator(self.sizes, Handle(self.u), Handle(self.v), num_nabla=2,
                                    calc_mode="calc_deriv", momentum_list=moms, ne_block_size=2).load("cfg")
        t = 1
        v = self.v[t].reshape(3,-1).T
        shifts = [reference_matrix(self.u, (d,), t) for d in range(6)]
        ds = [shifts[d] - shifts[d+3] for d in range(3)]
        actual_deriv = deriv.calc_path_blocks(t)["v2v"]
        written = np.empty_like(actual_deriv)
        self.assertIsNone(deriv.calc_path_blocks(t, recipe="derivatives",
            write=lambda ri,mi,se,sl,value: written.__setitem__((ri,mi,*sl),value)))
        for mi, mom in enumerate(moms):
            f = reference_matrix(self.u, (), t, mom)
            # Expand the two-sided definition using independent dense D matrices.
            for di, directions in enumerate(deriv.derivative_list):
                expected = np.zeros((3,3), complex)
                for assignments in itertools.product((0,1), repeat=len(directions)):
                    left, right = v.copy(), v.copy()
                    for d in [d for d,side in zip(directions,assignments) if not side][::-1]:
                        left = ds[d] @ left
                    for d in [d for d,side in zip(directions,assignments) if side]:
                        right = ds[d] @ right
                    expected += (-1)**sum(assignments) * left.conj().T @ f @ right
                np.testing.assert_allclose(actual_deriv[di,mi], expected, atol=8e-13, rtol=2e-13)
                np.testing.assert_allclose(written[di,mi], expected, atol=8e-13, rtol=2e-13)

    def test_fused_projection_matches_independent_path_sum(self):
        rows = [{"id":0,"terms":[{"path":[0,1],"coefficient":[.3,.8]},
                                   {"path":[3],"coefficient":[-.2,.7]}]},
                {"id":1,"terms":[{"path":[],"coefficient":[2,0]}]}]
        gen = self.gen([()], momentum_list=[(1,-1,0)])
        out = gen.calc_path_blocks(1,rows=rows)["v2v"]
        v = self.v[1].reshape(3,-1).T
        for ri, row in enumerate(rows):
            matrix = sum(complex(*term["coefficient"]) * reference_matrix(self.u, term["path"], 1, (1,-1,0)) for term in row["terms"])
            np.testing.assert_allclose(out[ri,0], v.conj().T @ matrix @ v, atol=8e-14)

    def test_projected_phase_residency_across_momentum_and_row_batches(self):
        rows = [
            {"id": 0, "terms": [
                {"path": [0, 1], "coefficient": [.3, .8]},
                {"path": [3], "coefficient": [-.2, .7]},
            ]},
            {"id": 1, "terms": [
                {"path": [], "coefficient": [1.1, -.4]},
            ]},
        ]
        momenta = [(index, 0, 0) for index in range(68)]
        estimate_size = (256 * 1024**2) // (2 * np.dtype(np.complex128).itemsize)
        self.assertEqual((256 * 1024**2) // (estimate_size * 16), 2)
        vectors = self.v[1, :3].reshape(3, -1).T

        for convention in ("midpoint", "anchor", "none"):
            with self.subTest(phase_convention=convention):
                gen = self.gen([()], momentum_list=momenta,
                               phase_convention=convention, is_blending=True,
                               dilution=((5,), (3,)))
                hinted_vectors = _EstimatedSizeArray(
                    gen._v(1, slice(0, gen.usedNe)), estimate_size)
                with mock.patch.object(gen, "_v", return_value=hinted_vectors):
                    borrowed = gen.calc_path_blocks(1,rows=rows)["v2v"]
                expected = borrowed.copy()

                for ri, row in enumerate(rows):
                    for mi, momentum in enumerate(momenta):
                        matrix = sum(
                            complex(*term["coefficient"])
                            * reference_matrix(self.u, term["path"], 1,
                                               momentum, convention)
                            for term in row["terms"]
                        )
                        value = vectors.conj().T @ matrix @ vectors
                        value *= gen.mode_pair_weights[:3, :3]
                        np.testing.assert_allclose(
                            expected[ri, mi], value, atol=2e-13, rtol=2e-13)

                borrowed[...] = -999.0
                with mock.patch.object(gen, "_v", return_value=hinted_vectors):
                    reused = gen.calc_path_blocks(1,rows=rows)["v2v"]
                self.assertIs(reused, borrowed)
                np.testing.assert_allclose(reused, expected, atol=2e-13, rtol=2e-13)

                streamed = self.gen([()], momentum_list=momenta,
                                    phase_convention=convention, is_blending=True,
                                    dilution=((5,), (3,)))
                streamed_vectors = _EstimatedSizeArray(
                    streamed._v(1, slice(0, streamed.usedNe)), estimate_size)
                original_phase = streamed._phase
                phase_refs = {}

                def track_phase(momentum, displacement=(0, 0, 0), convention=None):
                    phase = original_phase(momentum, displacement, convention)
                    phase_refs[tuple(momentum)] = weakref.ref(phase)
                    return phase

                callback_values = np.empty_like(expected)
                callback_order = []
                first_phase_alive_at_first_row_tail = []

                def write(ri, mi, sector, slices, value):
                    self.assertEqual(sector, "v2v")
                    self.assertEqual(slices, (slice(None), slice(None)))
                    callback_order.append((ri, mi))
                    callback_values[ri, mi] = np.array(value, copy=True)
                    if ri == 0 and mi == len(momenta) - 1:
                        first_phase_alive_at_first_row_tail.append(
                            phase_refs[momenta[0]]() is not None)

                with mock.patch.object(streamed, "_v", return_value=streamed_vectors), \
                     mock.patch.object(streamed, "_phase", side_effect=track_phase):
                    result = streamed.calc_path_blocks(1,rows=rows,write=write)

                self.assertIsNone(result)
                self.assertEqual(callback_order,
                                 [(ri, mi) for ri in range(len(rows))
                                  for mi in range(len(momenta))])
                self.assertEqual(first_phase_alive_at_first_row_tail, [False])
                np.testing.assert_allclose(callback_values, expected,
                                           atol=2e-13, rtol=2e-13)

                late_failure = self.gen([()], momentum_list=momenta,
                                        phase_convention=convention,
                                        is_blending=True, dilution=((5,), (3,)))
                late_hint = _EstimatedSizeArray(
                    late_failure._v(1, slice(0, late_failure.usedNe)), estimate_size)
                build_phase = late_failure._phase
                failed_prefix = []

                def fail_on_last_momentum(momentum, displacement=(0, 0, 0), convention=None):
                    if tuple(momentum) == momenta[-1]:
                        raise RuntimeError("injected late phase failure")
                    return build_phase(momentum, displacement, convention)

                with mock.patch.object(late_failure, "_v", return_value=late_hint):
                    with mock.patch.object(late_failure, "_phase",
                                           side_effect=fail_on_last_momentum):
                        with self.assertRaisesRegex(RuntimeError,
                                                    "injected late phase failure"):
                            late_failure.calc_path_blocks(1,rows=rows,write=lambda ri, mi, sector, slices, value:
                                failed_prefix.append((ri, mi)))
                    self.assertEqual(failed_prefix,
                                     [(0, mi) for mi in range(len(momenta) - 2)])
                    retried = late_failure.calc_path_blocks(1,rows=rows)["v2v"]
                np.testing.assert_allclose(retried, expected,
                                           atol=2e-13, rtol=2e-13)


    def test_cpu_projection_groups_paths_but_preserves_winding_and_zero_rows(self):
        from lqcdweave.production.elemental import projection
        rows = [dict(id=0,terms=[
                    dict(path=[0,1],coefficient=[.3,.2]),
                    dict(path=[1,0],coefficient=[-.4,.7]),
                    dict(path=[0,1],coefficient=[.1,-.3])]),
                dict(id=1,terms=[dict(path=[],coefficient=[.5,.2]),
                                 dict(path=[],coefficient=[-.5,-.2])]),
                dict(id=2,terms=[dict(path=[0],coefficient=[.4,-.2]),
                                 dict(path=[0,0,0,0],coefficient=[.2,.3])])]
        momenta = [(0,0,0),(1,-1,0),(-1,0,1)]
        vectors = self.v[1].reshape(3,-1).T
        for convention in ('midpoint','anchor','none'):
            for cache_bytes in (0,512*1024**2):
                with self.subTest(phase=convention,cache_bytes=cache_bytes):
                    gen = self.gen([()],momentum_list=momenta,phase_convention=convention,
                                   is_blending=True,dilution=((5,),(3,)))
                    field_bytes = self.v[1].nbytes
                    with mock.patch.object(projection,'_TRANSPORT_CACHE_BYTES',cache_bytes), \
                         mock.patch.object(projection,'_WEIGHT_BYTES',2*field_bytes), \
                         mock.patch.object(projection,'apply_transport',wraps=projection.apply_transport) as apply:
                        actual = gen.calc_path_blocks(1,rows=rows)['v2v'].copy()
                    # Three nonzero row/endpoint groups, each used in two momentum batches.
                    self.assertEqual(apply.call_count,6)
                    for ri,row in enumerate(rows):
                        for mi,momentum in enumerate(momenta):
                            matrix = sum(complex(*term['coefficient'])*reference_matrix(
                                self.u,term['path'],1,momentum,convention) for term in row['terms'])
                            expected = (vectors.conj().T @ matrix @ vectors)*gen.mode_pair_weights
                            np.testing.assert_allclose(actual[ri,mi],expected,atol=3e-13,rtol=3e-13)
                    np.testing.assert_array_equal(actual[1],0)

    def test_cpu_projection_does_not_cache_numeric_links_between_calls(self):
        rows = [dict(id=0,terms=[dict(path=[0,1],coefficient=[.3,.8]),
                                 dict(path=[1,0],coefficient=[-.4,.2])])]
        gen = self.gen([()],momentum_list=[(1,-1,0)])
        before = gen.calc_path_blocks(0,rows=rows)['v2v'].copy()
        gen.stout_smear(1,.12)
        after = gen.calc_path_blocks(0,rows=rows)['v2v'].copy()
        self.assertGreater(np.linalg.norm(after-before),1e-4)
        vectors = self.v[0].reshape(3,-1).T
        links = stout_smear(self.u.transpose(4,0,1,2,3,5,6),1,.12).transpose(1,2,3,4,0,5,6)
        matrix = sum(complex(*term['coefficient'])*reference_matrix(
            links,term['path'],0,(1,-1,0)) for term in rows[0]['terms'])
        np.testing.assert_allclose(after[0,0],vectors.conj().T @ matrix @ vectors,atol=3e-13,rtol=3e-13)
        gen.load('new-cfg')
        np.testing.assert_allclose(gen.calc_path_blocks(0,rows=rows)['v2v'],before,atol=3e-13,rtol=3e-13)

    def test_dense_projection_tiles_right_modes_without_retransporting_each_momentum(self):
        from lqcdweave.production.elemental import projection
        rows = [dict(id=0, terms=[dict(path=[0], coefficient=[.3, .7]),
                                  dict(path=[1, 0], coefficient=[-.2, .4])])]
        momenta = [(i, -1, 0) for i in range(5)]
        vectors = self.v[1].reshape(3, -1).T
        budget = 4 * self.v[1].nbytes
        for convention in ('midpoint', 'anchor', 'none'):
            for cache in (0, 512*1024**2):
                with self.subTest(convention=convention, cache=cache):
                    gen = self.gen([()], momentum_list=momenta, phase_convention=convention,
                                   is_blending=True, dilution=((5,), (3,)))
                    zeros = np.zeros
                    weighted_allocations = []

                    def allocate(shape, *args, **kwargs):
                        if len(shape) == 7:
                            weighted_allocations.append(int(np.prod(shape))*16)
                        return zeros(shape, *args, **kwargs)

                    with mock.patch.object(projection, '_WEIGHT_BYTES', budget), \
                         mock.patch.object(projection, '_TRANSPORT_CACHE_BYTES', cache), \
                         mock.patch.object(projection, 'apply_transport', wraps=projection.apply_transport) as transport, \
                         mock.patch.object(np, 'zeros', side_effect=allocate):
                        actual = gen.calc_path_blocks(1, rows=rows)['v2v'].copy()
                    # Both endpoint groups transport exactly Ne modes, once for all momenta.
                    self.assertEqual(sum(call.args[1].shape[0] for call in transport.call_args_list), 2*3)
                    self.assertTrue(weighted_allocations)
                    self.assertLessEqual(max(weighted_allocations), budget)
                    for mi, momentum in enumerate(momenta):
                        matrix = sum(complex(*term['coefficient'])*reference_matrix(
                            self.u, term['path'], 1, momentum, convention) for term in rows[0]['terms'])
                        expected = (vectors.conj().T @ matrix @ vectors)*gen.mode_pair_weights
                        np.testing.assert_allclose(actual[0, mi], expected, atol=3e-13, rtol=3e-13)

    def test_baryon_epsilon_and_blocking(self):
        fields = (self.v[0,:2], self.v[1,:3], self.v[2,:1])
        expected = np.zeros((2,3,1), complex)
        for z,y,x in np.ndindex(*self.v.shape[2:5]):
            phase = np.exp(2j*np.pi*x/self.sizes[0])
            for i,j,k in np.ndindex(expected.shape):
                # Determinant supplies an independent epsilon contraction.
                expected[i,j,k] += phase*np.linalg.det(np.stack([fields[0][i,z,y,x],fields[1][j,z,y,x],fields[2][k,z,y,x]]))
        for batch in (1,5,100):
            actual = generate_baryon_elemental(fields, (1,0,0), self.sizes[:3], block_size=batch)
            np.testing.assert_allclose(actual, expected, atol=8e-14)

    def test_gauge_preparation_batches_and_independent_staples(self):
        links = self.u.transpose(4,0,1,2,3,5,6)
        original = links.copy()
        result = stout_smear(links, 1, .08, matrix_batch_size=5)
        # Independent coordinate loop, including both staples in each plane.
        expected = links.copy()
        lt,lz,ly,lx = self.u.shape[:4]
        def u(mu,t,p):
            x,y,z = p % np.array([lx,ly,lz])
            return self.u[t,z,y,x,mu]
        eye = np.eye(3)
        for t,z,y,x in np.ndindex(lt,lz,ly,lx):
            p = np.array([x,y,z])
            for mu in range(3):
                staple = np.zeros((3,3),complex)
                for nu in range(3):
                    if mu == nu:
                        continue
                    em,en = np.eye(3,dtype=int)[[mu,nu]]
                    staple += u(nu,t,p)@u(mu,t,p+en)@u(nu,t,p+em).conj().T
                    staple += u(nu,t,p-en).conj().T@u(mu,t,p-en)@u(nu,t,p-en+em)
                omega = .08*staple@u(mu,t,p).conj().T
                q = .5j*(omega.conj().T-omega)
                q -= np.trace(q)*eye/3
                values,vectors = np.linalg.eigh(q)
                expected[mu,t,z,y,x] = (vectors@np.diag(np.exp(1j*values))@vectors.conj().T)@u(mu,t,p)
        np.testing.assert_allclose(result,expected,atol=3e-15)
        np.testing.assert_array_equal(links,original)
        np.testing.assert_array_equal(result[3],links[3])
        projected = project_su3(links * (1.1+.3j),matrix_batch_size=7)
        np.testing.assert_allclose(projected@projected.conj().swapaxes(-1,-2),np.broadcast_to(eye,projected.shape),atol=2e-15)
        np.testing.assert_allclose(np.linalg.det(projected),1,atol=2e-15)

    def test_complex_recombination_and_pair_weights_once(self):
        array = np.arange(24).reshape(2,3,4).astype(complex)
        rows = [[(.3+.7j,0),(-2j,2)],[(1,1),(2,1)]]
        out = np.empty((2,2,4),complex)
        self.assertIs(re_combine(array,rows,1,out),out)
        np.testing.assert_allclose(out[:,0],(.3+.7j)*array[:,0]-2j*array[:,2])
        np.testing.assert_allclose(out[:,1],3*array[:,1])
        with self.assertRaises(ValueError):
            re_combine(array,[[(1,0)],[(1,1)],[(1,2)]],1,array)
        # Enumerate every sample: inverse inclusion weights recover all pairs.
        weights = dilution_pair_weights([4],[2])
        accumulated = np.zeros((4,4))
        samples = list(itertools.combinations(range(4),2))
        for sample in samples:
            accumulated[np.ix_(sample,sample)] += weights/len(samples)
        np.testing.assert_allclose(accumulated,1)
        gen = self.gen([()],is_blending=True,dilution=([6],[3]))
        base = self.gen([()]).calc_path_blocks(0)["v2v"]
        np.testing.assert_allclose(gen.calc_path_blocks(0)["v2v"],base*dilution_pair_weights([6],[3]))

    def test_sparse_point_dense_conversion_rejects_duplicate_pairs(self):
        from lqcdweave.contracts.elemental import SparsePointBlock
        block = SparsePointBlock(np.array([0,1]),np.array([1,0]),np.arange(18).reshape(2,3,3),(2,2))
        dense=block.to_dense()
        np.testing.assert_array_equal(dense[0,:,1,:],block.values[0])
        np.testing.assert_array_equal(dense[1,:,0,:],block.values[1])
        duplicate=SparsePointBlock(np.array([0,0]),np.array([1,1]),np.arange(18).reshape(2,3,3),(2,2))
        with self.assertRaisesRegex(ValueError,"duplicate point pairs"):
            duplicate.to_dense()


if __name__ == "__main__":
    unittest.main()
