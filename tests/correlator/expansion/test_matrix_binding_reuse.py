"""Binding-local memoization preserves actual endpoint and preparation contracts."""
from collections import Counter
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_structured_correlator import fixture, task
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from correlator.execution.test_cfg_runtime import SafeReader
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.correlator.api import DiagramBindings, describe_coordinates
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation
from lqcdweave.contraction.expansion.api import MatrixCatalogBinder, describe_meson_matrix
from lqcdweave.contraction.expansion.physical_inputs import PhysicalInputRegistry
from lqcdweave.operators.correlator.api import (
    Flavor, Hadron, HadronIrrep, Tag, gen_correlator,
    remove_disconneted_diagram, split_meson_mode_pairs,
)


class MatrixBindingReuseTests(unittest.TestCase):
    def setUp(self):
        self.bind, self.elemental, self.propagator = fixture()
        self.bindings = DiagramBindings(vertex_map=self.bind, configuration='cfg', usedNe=3,
            propagator_map={'S^q': self.propagator, r'S^q_\mathrm{local}': self.propagator})

    def request(self):
        full = split_meson_mode_pairs(task(structured=True, flavor='I0'), partitions=(0, 1, 3))
        return {'full': full, 'copy': full, 'disconnected': full-remove_disconneted_diagram(
            full, [r'S^q_\mathrm{local}'])}

    def test_each_actual_probe_and_endpoint_registration_once_per_coordinate_binding(self):
        described = describe_meson_matrix(self.request(), bindings=self.bindings)
        binder = MatrixCatalogBinder(described, bindings=self.bindings, _factorize_spin=True)
        preparation = Meson.describe_factor_preparation
        register = PhysicalInputRegistry.register_propagator
        prior = None
        for times in ({0: 1, 1: 0}, {0: 0, 1: 1}):
            probes, edges = Counter(), Counter()
            def describe(meson, factor, block, **options):
                if block == 'unbound':
                    identity = id(meson), id(factor)
                    probes[identity] += 1
                return preparation(meson, factor, block, **options)
            def edge(registry, reader, symbol, source, sink, sector, **options):
                identity = id(reader), str(symbol), id(source), id(sink), sector, tuple(options.items())
                edges[identity] += 1
                return register(registry, reader, symbol, source, sink, sector, **options)
            coordinates, _ = describe_coordinates(time_axes=times)
            with patch.object(Meson, 'describe_factor_preparation', describe), \
                    patch.object(PhysicalInputRegistry, 'register_propagator', edge), \
                    patch.object(self.elemental, 'block', side_effect=AssertionError('early IO')):
                expanded = binder.bind(coordinates=coordinates)
            self.assertTrue(probes)
            self.assertEqual(set(probes.values()), {1})
            self.assertTrue(edges)
            self.assertEqual(set(edges.values()), {1})
            # Rebinding the same binder must describe current physical times,
            # with a fresh registry and loader rather than reusing old sources.
            arguments = tuple((key, r.arguments) for key, r in expanded.inputs.read_requests.items())
            if prior is not None:
                self.assertNotEqual(arguments, prior)
            prior = arguments

    def test_unit_spin_probe_is_bounded_by_actual_component_selection(self):
        import lqcdweave.contraction.expansion.matrix_inputs as matrix_inputs
        described = describe_meson_matrix(self.request(), bindings=self.bindings)
        original = matrix_inputs.unit_spin
        probes = []
        def unit(recipe):
            probes.append(recipe)
            return original(recipe)
        binder = MatrixCatalogBinder(described, bindings=self.bindings, _factorize_spin=True)
        for times in ({0: 1, 1: 0}, {0: 0, 1: 1}):
            probes.clear()
            coordinates, _ = describe_coordinates(time_axes=times)
            with patch.object(matrix_inputs, 'unit_spin', unit):
                expanded = binder.bind(coordinates=coordinates)
            self.assertTrue(probes)
            # The same actual recipe is tested once even when many Wick jobs
            # consume it. A new coordinate binding must test its new recipes.
            self.assertEqual(len(probes), len({id(recipe) for recipe in probes}))
            self.assertTrue(expanded.problem.terms)

    def test_nonmonomial_fallback_is_cached_and_retains_dense_recipes(self):
        described = describe_meson_matrix(self.request(), bindings=self.bindings)
        original = Meson.describe_factor_preparation
        probes = Counter()
        def nonmonomial(meson, description, block, **options):
            recipe = original(meson, description, block, **options)
            # Actual finite coefficients now have two nonzeros per spin row.
            spin = recipe.constants[0].array().copy()
            spin += np.roll(spin, 1, axis=1)*.5
            from lqcdweave.contracts.correlator.api import ArrayConstant
            if block == 'unbound':
                probes[id(meson), id(description)] += 1
            return replace(recipe, constants=(ArrayConstant.from_array(spin), *recipe.constants[1:]))
        with patch.object(Meson, 'describe_factor_preparation', nonmonomial):
            factorized = MatrixCatalogBinder(described, bindings=self.bindings,
                                             _factorize_spin=True).bind()
            ordinary = MatrixCatalogBinder(described, bindings=self.bindings).bind()
        self.assertTrue(probes)
        self.assertEqual(set(probes.values()), {1})
        self.assertEqual(factorized.problem, ordinary.problem)
        self.assertEqual(factorized.output_recipe, ordinary.output_recipe)
        self.assertEqual(factorized.inputs.read_requests, ordinary.inputs.read_requests)
        self.assertFalse(any(p.recipe.spin_columns for p in factorized.inputs.loader.preparations.values()))

    def test_distinct_flavor_symbols_keep_aliases_and_actual_readers(self):
        row = HadronIrrep('z', (0, 0, 0), 'A_1', -1, Tag(0, 0))[0]
        axis = [Hadron(row, Flavor('uu')+Flavor('dd'))]
        request = gen_correlator([axis.copy(), axis.copy()], [0, 1], [False, True],
                                 structured=True, degenerate=False)
        def expand(second):
            propagators = {key: reader for flavor, reader in (('u', self.propagator), ('d', second))
                for key in (f'S^{flavor}', rf'S^{flavor}_\mathrm{{local}}')}
            bindings = replace(self.bindings, propagator_map=propagators)
            described = describe_meson_matrix(request, bindings=bindings,
                                               representative_times={0: 1, 1: 0})
            return MatrixCatalogBinder(described, bindings=bindings, _factorize_spin=True).bind()
        shared = expand(self.propagator)
        roles = {r.reader_role for primary in shared.inputs.read_requests.values()
                 for r in (primary, *primary.aliases) if r.family == 'propagator'}
        self.assertTrue({'propagator:S^u', 'propagator:S^d'} <= roles)
        other = ArrayPropagator({key: value*(.5+1j) for key, value in self.propagator.sectors.items()},
            time_convention='absolute', cfg='cfg', data_identity='independent-other-flavor')
        separate = expand(other)
        identities = {r.metadata.data_identity for r in separate.inputs.read_requests.values()
                      if r.family == 'propagator'}
        self.assertEqual(identities, {self.propagator.data_identity, other.data_identity})

    def test_public_pc_numerics_and_two_cfg_fixed_archive_replay(self):
        options = dict(vertex_map=self.bind, propagator_map=self.bindings.propagator_map,
                       configuration='cfg', usedNe=3, memory_limit=1<<28, host_memory_limit=1<<28)
        request = self.request()
        selected = calculate(request, **options, optimize='meson_matrix', execute=False,
            time_axes={0: 1, 1: 0}, search={'deadline_seconds': 1., 'max_states': 1000,
                                         'component_workers': 1})
        pairs = [{0: 1, 1: 0}, {0: 0, 1: 1}]
        coordinates, _ = describe_coordinates(time_pairs=pairs)
        complete = complete_calculation(selected, coordinates=coordinates)
        expected = {}
        for cfg, scale in (('a', 1), ('b', 2)):
            e = ArrayElemental({key: value*scale for key, value in self.elemental.sectors.items()},
                self.elemental.components, phase_convention='anchor', cfg=cfg,
                data_identity=self.elemental.data_identity+':'+cfg)
            p = ArrayPropagator({key: value*scale for key, value in self.propagator.sectors.items()},
                time_convention='absolute', cfg=cfg, data_identity=self.propagator.data_identity+':'+cfg)
            binding = lambda row: self.bind(row).with_reader(e, configuration=cfg)
            expected[cfg] = calculate(request, vertex_map=binding,
                propagator_map={'S^q': p, r'S^q_\mathrm{local}': p}, configuration=cfg,
                usedNe=3, memory_limit=1<<28, host_memory_limit=1<<28,
                time_pairs=pairs, optimize='none').value
        @contextmanager
        def factory(cfg):
            scale = 1 if cfg == 'a' else 2
            yield {r.reader_role: SafeReader(self.elemental if r.family == 'elemental'
                    else self.propagator, cfg, scale, [])
                for primary in complete.plan.requests.values() for r in (primary, *primary.aliases)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'complete.plan.gz'
            save_calculation_plan(path, complete)
            with ExitStack() as blocked:
                for name in ('lqcdweave.contraction.correlator.expand_correlator',
                             'lqcdweave.contraction.expansion.matrix_inputs._bind_matrix_base',
                             'lqcdweave.contraction.correlator.find_contraction_path',
                             'lqcdweave.contraction.planning.cfg_plan._bind_cfg_coordinates'):
                    blocked.enter_context(patch(name, side_effect=AssertionError(name)))
                values = dict(calculate(plan_archive=path, configuration=iter(('a', 'b')),
                                        data=factory).value)
        for cfg in expected:
            for name in request:
                np.testing.assert_allclose(values[cfg][name], expected[cfg][name], rtol=3e-12, atol=3e-12)


if __name__ == '__main__':
    unittest.main()
