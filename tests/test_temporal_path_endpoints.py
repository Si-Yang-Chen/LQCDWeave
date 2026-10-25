"""Ordinary temporal paths: physical times and segmented pools, independent sum."""

from itertools import product
from math import prod
import unittest

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.dilution import DilutionGroup, DilutionSource
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram, Propagator, Tag
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.contraction.correlator import calculate


class TemporalPathEndpointTests(unittest.TestCase):
    def test_plain_temporal_d_paths_use_physical_time_pools(self):
        rng = np.random.default_rng(752)
        component = ElementalComponent.path((6,))
        operator = Operator.from_terms('temporal-D', (OperatorTerm(1, component, 0),))
        pools = ('first', 'first', 'second', 'second')
        group = DilutionGroup((0, 1, 2), ((0, 0), (0, 1)))
        source = DilutionSource('actual-D', 'parent', (group,), tuple(enumerate(pools)))
        raw = rng.normal(size=(1, 1, 4, 2, 2)) + 1j * rng.normal(size=(1, 1, 4, 2, 2))
        values = rng.normal(size=(4, 4, 4, 4, 2, 2)) + 1j * rng.normal(size=(4, 4, 4, 4, 2, 2))
        reader = ArrayPropagator({'dsd': values}, time_convention='absolute', cfg='temporal-D',
            basis_endpoints={('dsd', side): (source.data_id, (0, 1)) for side in ('sink', 'source')})
        symbols = (Propagator('u', Tag(0, 0), Tag(1, 0)), Propagator('u', Tag(1, 0), Tag(0, 0)))
        diagram = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 1], ['a', 'b'], [None, *symbols])
        pair_weights = np.where(np.eye(2, dtype=bool), 1.5, 3.)
        weighted_raw = raw.copy()
        for t in range(4):
            weighted_raw[0, 0, t] *= pair_weights if pools[t] == pools[(t+1) % 4] else 2.25
        for weighted in (False, True):
            elemental = ArrayElemental({'d2d': weighted_raw if weighted else raw}, (component,),
                cfg='temporal-D', phase_convention='anchor', elemental_weighted=weighted,
                basis_endpoints={('d2d', side): (source.data_id, (0, 1)) for side in ('bar', 'field')},
                dilution_endpoints={('d2d', side): source for side in ('bar', 'field')})
            vertices = {name: Meson(elemental, operator, requested_sectors=('d2d',), phase_convention='anchor') for name in ('a', 'b')}
            # Anchors 0,2 have physical endpoint pairs (0,1),(2,3). D's
            # two explicit shared pools each contribute one pair probability.
            s1 = values[2, 1].transpose(0, 2, 1, 3)
            s2 = values[0, 3].transpose(0, 2, 1, 3)
            tensor = np.einsum('ubwc,wdua->abcd', np.einsum('uv,vbwc->ubwc', gamma(1), s1),
                               np.einsum('uv,vbwc->ubwc', gamma(1), s2))
            expected = np.einsum('ab,cd,abcd,ab,cd->', raw[0, 0, 0], raw[0, 0, 2], tensor, pair_weights, pair_weights)
            result = calculate(diagram, vertex_map=vertices, propagator_map={s: reader for s in symbols},
                configuration='temporal-D', time_axes={0: (0,), 1: (2,)}, groups_independent=True,
                elemental_weighted=weighted, optimize='none', profile='default')
            np.testing.assert_allclose(result.value.item(), expected, atol=2e-9, rtol=2e-10)

    def test_plain_paths_joint_point_sum_raw_weighted_dagger_and_wrap(self):
        rng = np.random.default_rng(362)
        lt, n, population = 4, 4, 6
        def random(shape):
            return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / 10
        raw = random((1, 1, lt, n, 3, n, 3))
        propagators = random((lt, lt, 4, 4, n, 3, n, 3))
        reader = ArrayPropagator({'psp': propagators}, time_convention='absolute', cfg='temporal-path')
        symbols = (Propagator('u', Tag(0, 0), Tag(1, 0)), Propagator('u', Tag(1, 0), Tag(0, 0)))
        diagram = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 1], ['a', 'b'], [None, *symbols])
        pools = ('first', 'first', 'second', 'second')
        groups = {p: SamplingPoolSpec(p, population, n) for p in set(pools)}
        for step, displacement in ((6, 1), (7, -1)):
            component = ElementalComponent.path((step,))
            coefficient = .7 + .3j
            operator = Operator.from_terms('plain-temporal-path', (OperatorTerm(1, component, 0, coefficient),))
            pool_ids = {(component, 0, t, side): pools[t] for t in range(lt) for side in ('bar', 'field')}
            weighted_values = raw.copy()
            for t in range(lt):
                endpoint = (t + displacement) % lt
                w = (np.where(np.eye(n, dtype=bool), population / n,
                              population * (population - 1) / (n * (n - 1)))
                     if pools[t] == pools[endpoint] else np.full((n, n), (population / n)**2))
                weighted_values[0, 0, t] *= w[:, None, :, None]
            for weighted, dagger, pair in product((False, True), (False, True), ((0, 2), (3, 0))):
                with self.subTest(step=step, weighted=weighted, dagger=dagger, pair=pair):
                    elemental = ArrayElemental({'p2p': weighted_values if weighted else raw}, (component,),
                        cfg='temporal-path', phase_convention='anchor', elemental_weighted=weighted,
                        sampling_pool_ids=pool_ids, sampling_groups=groups)
                    vertices = {'a': Meson(elemental, operator, source=dagger, requested_sectors=('p2p',), phase_convention='anchor'),
                                'b': Meson(elemental, operator, requested_sectors=('p2p',), phase_convention='anchor')}
                    t, u = pair
                    a, b, c, d = t, (t + displacement) % lt, u, (u + displacement) % lt
                    e1 = raw[0, 0, t].reshape(3*n, 3*n) * coefficient
                    e2 = raw[0, 0, u].reshape(3*n, 3*n) * coefficient
                    g1, g2 = gamma(1), gamma(1)
                    if dagger:
                        a, b = b, a
                        e1 = e1.conj().T
                        g1 = gamma(8) @ g1.conj().T @ gamma(8)
                    # Storage [source,sink], physical matrix [sink spin/point,source spin/point].
                    s1 = propagators[c, b].transpose(0, 2, 3, 1, 4, 5).reshape(4, 3*n, 4, 3*n)
                    s2 = propagators[a, d].transpose(0, 2, 3, 1, 4, 5).reshape(4, 3*n, 4, 3*n)
                    tensor = np.einsum('ubwc,wdua->abcd', np.einsum('uv,vbwc->ubwc', g1, s1),
                                       np.einsum('uv,vbwc->ubwc', g2, s2))
                    weights = np.empty((n,)*4)
                    owners = tuple(pools[x] for x in (a, b, c, d))
                    for labels in np.ndindex(weights.shape):
                        weight = 1.
                        for pool in dict.fromkeys(owners):
                            r = len({label for label, owner in zip(labels, owners) if owner == pool})
                            weight *= prod(population-i for i in range(r)) / prod(n-i for i in range(r))
                        weights[labels] = weight
                    for axis in range(4):
                        weights = weights.repeat(3, axis=axis)
                    expected = np.einsum('ab,cd,abcd,abcd->', e1, e2, tensor, weights)
                    result = calculate(diagram, vertex_map=vertices, propagator_map={s: reader for s in symbols},
                        configuration='temporal-path', time_axes={0: (t,), 1: (u,)}, groups_independent=True,
                        elemental_weighted=weighted, optimize='none', profile='default')
                    np.testing.assert_allclose(result.value.item(), expected, atol=2e-11, rtol=2e-10)


if __name__ == '__main__':
    unittest.main()
