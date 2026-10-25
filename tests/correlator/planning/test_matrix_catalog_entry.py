"""Public structured calculations retain whole vertices and fixed replay."""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from test_structured_correlator import fixture, task
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.planning.api import find_contraction_path
import lqcdweave.contraction.planning.path_planning as path_planning
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation


def bounded_search():
    return {'deadline_seconds': 1.0, 'max_states': 1000, 'component_workers': 1}


class MatrixCatalogEntryTests(unittest.TestCase):
    def setUp(self):
        self.bind, self.elemental, self.propagator = fixture()
        self.options = dict(vertex_map=self.bind, configuration='cfg', usedNe=3,
            propagator_map={'S^q':self.propagator, r'S^q_\mathrm{local}':self.propagator},
            memory_limit=1<<30, host_memory_limit=1<<30)

    def test_matrix_entry_preserves_whole_vertices_and_independent_values(self):
        request = task(structured=True, flavor='I0')
        for coordinates in ({}, dict(time_axes={0:1, 1:0})):
            with self.subTest(coordinates=coordinates):
                options = dict(self.options, partitions=(0,1,2,3), **coordinates)
                with (patch('lqcdweave.contraction.expansion.structured._Compiler.task',
                            side_effect=AssertionError('scalar construction')),
                      patch.object(path_planning, 'solve_storage',
                                   wraps=path_planning.solve_storage) as search,
                      patch.object(self.elemental, 'read_slab', side_effect=AssertionError('early IO'))):
                    result = calculate(request, optimize='meson_matrix', execute=False,
                                       search=bounded_search(), **options)
                search.assert_not_called()
                self.assertFalse(result.prediction['planner']['search_performed'])
                self.assertIsNone(result.search)
                self.assertTrue(result.plan.state.actions)
                self.assertGreaterEqual(result.plan.reserved_host_bytes, result.inputs.assembly_host_bytes)
                reference = calculate(task(structured=False,flavor='I0'), optimize='none', **options).value
                np.testing.assert_allclose(result.run().value, reference, rtol=3e-12, atol=3e-12)
                np.testing.assert_allclose(result.run(measure=True).value, reference, rtol=3e-12, atol=3e-12)

    def test_domain_and_budget_rejections_before_reading(self):
        request = task(structured=True)
        expanded = calculate(request, optimize='meson_matrix', execute=False,
                             search=bounded_search(), **self.options).expanded
        with patch.object(self.elemental, 'read_slab', side_effect=AssertionError('early IO')):
            with patch.object(path_planning, 'solve_storage',
                              wraps=path_planning.solve_storage) as search:
                path = find_contraction_path(expanded, memory_limit=1<<30,
                    optimize='meson_matrix', search=bounded_search())
            search.assert_not_called()
            self.assertFalse(path.prediction['planner']['search_performed'])
            self.assertTrue(path.plan.state.actions)
            with self.assertRaisesRegex(MemoryError, 'storage budget'):
                calculate(request, optimize='meson_matrix', execute=False,
                          search=bounded_search(),
                          **dict(self.options, memory_limit=128, host_memory_limit=128))
            bounded = calculate(request, optimize='meson_matrix', execute=False,
                search={'deadline_seconds': 1.0, 'max_states': 1000, 'root_width': 4, 'component_workers': 1},
                **self.options)
            self.assertFalse(bounded.prediction['planner']['search_performed'])
            self.assertTrue(bounded.plan.state.actions)

    def test_selected_matrix_path_completes_and_loads_from_raw_elemental_reader(self):
        selected = calculate(task(structured=True), optimize='meson_matrix',
            time_axes={0:0,1:1}, partitions=(0,1,3), execute=False,
            search=bounded_search(),
            target_hardware=TargetHardwareMemory(host_available_bytes=1<<30), **self.options)
        pairs = ({0:0,1:1},{0:1,1:0})
        coordinates, _ = describe_coordinates(time_pairs=pairs)
        result = complete_calculation(selected,coordinates=coordinates,
            readers=selected.inputs.loader.input_readers)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'matrix.json.gz'
            save_calculation_plan(archive, result)
            with (patch('lqcdweave.contraction.correlator.expand_correlator',
                        side_effect=AssertionError('expansion')),
                  patch('lqcdweave.contraction.correlator.find_contraction_path',
                        side_effect=AssertionError('search')),
                  patch('lqcdweave.contraction.planning.api.audit_cfg_plan',
                        side_effect=AssertionError('resource planning'))):
                restored = calculate(None,plan_archive=archive,configuration='cfg',
                    propagator_map=self.options['propagator_map'],vertex_map=self.elemental,
                    memory_limit=1<<30,host_memory_limit=1<<30,execute=False)
                actual=restored.run().value
            reference = calculate(task(flavor='charged'),optimize='none',time_pairs=pairs,
                                  partitions=(0,1,3),**self.options).value
            np.testing.assert_allclose(actual,reference,rtol=3e-12,atol=3e-12)
            self.assertEqual(len(restored.plan.instances),len(pairs))

    def test_non_numpy_provider_metadata_uses_unified_search_without_reads(self):
        other_backend = SimpleNamespace(__name__='cupy')
        def other_vertex(row):
            vertex = self.bind(row)
            vertex.xp = other_backend
            return vertex
        with patch.object(self.elemental, 'read_slab', side_effect=AssertionError('early IO')):
            result = calculate(task(structured=True), optimize='meson_matrix', execute=False,
                search=bounded_search(), **dict(self.options,vertex_map=other_vertex))
        self.assertTrue(result.prediction['planner']['search_performed'])
        self.assertTrue(result.plan.state.actions)


if __name__ == '__main__':
    unittest.main()
