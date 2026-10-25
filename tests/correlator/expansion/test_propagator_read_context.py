"""Primary-source data stays explicit through one joint fixed replay."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.operators.correlator.api import Flavor, Hadron, HadronIrrep, Tag, gen_correlator
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm


class ContextReader(ArrayPropagator):
    read_context_axes = ('source',)
    describe_slab = read_slab = None
    background_read_safe = False

    def __init__(self, raw, selection):
        self.raw = raw
        self.reads = []
        self.context_readers = []
        for primary in range(len(selection.times)):
            blocks = {(t, t): raw[t, 0] for t in range(len(selection.times))}
            for sink in range(len(selection.times)):
                if sink != primary:
                    block = raw[primary, (sink-primary) % len(selection.times)]
                    blocks[sink, primary] = block
                    # Synthetic non-Hermitian source rows, deliberate reconstruction.
                    blocks[primary, sink] = block.conj().swapaxes(0, 1)
            self.context_readers.append(ArrayPropagator({'vsv': blocks}, time_convention='absolute',
                cfg='cfg', data_identity='raw-solves', source_selection=selection, sink_selection=selection))
        self.__dict__.update({key: value for key, value in self.context_readers[0].__dict__.items()
                              if key not in {'context_readers', 'raw', 'reads'}})

    def describe_block(self, sink, source, sector='vsv', primary=None, **selection):
        metadata = self.context_readers[primary].describe_block(sink, source, sector, **selection)
        # The same endpoint pair from distinct primary solves is distinct input.
        return replace(metadata, selection_identity=repr((primary, metadata.selection_identity)))

    def block(self, sink, source, sector='vsv', primary=None, **selection):
        self.reads.append((primary, sink, source))
        return self.context_readers[primary].block(sink, source, sector, **selection)


def fixture(seed=2):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(3, 3, 4, 4, 1, 1)) + 1j*rng.normal(size=(3, 3, 4, 4, 1, 1))
    selection = TimeSelectionRecord(128, (0, 1, 127))
    reader = ContextReader(raw, selection)
    component = ElementalComponent(0)
    elemental = ArrayElemental({'v2v': np.ones((1, 1, 3, 1, 1), dtype=complex)}, (component,),
        cfg='cfg', phase_convention='anchor', data_identity='phi', selection=selection)
    operator = Operator.from_terms('p', (OperatorTerm(0, component, 0),))
    vertices = {dagger: Meson(elemental, operator, dagger, phase_convention='anchor')
                for dagger in (False, True)}
    row = HadronIrrep('p', (0, 0, 0), 'A_1', 1, Tag(0, 0))[0]
    channels = [Hadron(row, Flavor('ud'))]
    task = gen_correlator([channels, channels], ['source', 'sink'], [True, False], degenerate=True)
    return task, reader, elemental, lambda row: vertices[row.dagger]


class PropagatorReadContextTests(unittest.TestCase):
    def options(self, reader, vertices):
        return dict(propagator_map={'S^q': reader, 'S^q_\\mathrm{local}': reader}, vertex_map=vertices,
            configuration='cfg', usedNe=1, optimize='none', profile=CostProfile(),
            target_hardware=TargetHardwareMemory(host_available_bytes=1 << 28),
            host_memory_limit=1 << 28, memory_limit=1 << 28)

    def test_one_plan_archives_all_contexts_and_replays_changed_data_without_ep(self):
        task, reader, elemental, vertices = fixture()
        selected = calculate(task, time_axes={'source': 0, 'sink': 1},
                             execute=False, **self.options(reader, vertices))
        coordinates, _ = describe_coordinates(time_axes={'source': range(3), 'sink': range(3)})
        selected = complete_calculation(selected, coordinates=coordinates,
            readers=selected.inputs.loader.input_readers)
        requests = tuple(selected.plan.requests.values())
        propagators = [request for request in requests if request.family == 'propagator']
        self.assertTrue(propagators)
        self.assertTrue(all(len(request.arguments) == 4 for request in propagators))
        self.assertEqual({request.arguments[3] for request in propagators}, {0, 1, 2})
        self.assertTrue(all(('read_context_axes', ('source',)) in request.conventions for request in propagators))
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'context.json.gz'
            save_calculation_plan(archive, selected)
            for seed in (2, 11):
                _, current, current_elemental, _ = fixture(seed)
                with (patch('lqcdweave.contraction.correlator.expand_correlator', side_effect=AssertionError('E')),
                      patch('lqcdweave.contraction.correlator.find_contraction_path', side_effect=AssertionError('P')),
                      patch('lqcdweave.contraction.planning.cfg_plan.build_cfg_plan', side_effect=AssertionError('schedule'))):
                    restored = calculate(plan_archive=archive, configuration='cfg',
                        propagator_map={'S^q': current, 'S^q_\\mathrm{local}': current},
                        vertex_map=current_elemental, execute=False)
                    actual = restored.run().value
                expected = []
                for source in range(3):
                    for sink in range(3):
                        a = current.context_readers[source].block(sink, source)[:, :, 0, 0]
                        b = current.context_readers[source].block(source, sink)[:, :, 0, 0]
                        expected.append(-np.einsum('ab,ba->', a, b))
                np.testing.assert_allclose(np.asarray(actual).reshape(-1), expected, rtol=2e-13, atol=2e-13)
                self.assertIn((0, 1, 0), current.reads)
                self.assertIn((1, 1, 0), current.reads)
                self.assertIn((2, 0, 2), current.reads)
                current.read_context_axes = ('sink',)
                with self.assertRaisesRegex(ValueError, 'conventions'):
                    restored.run()

    def test_fixed_context_slab_callbacks_and_batches_keep_primary_source(self):
        class SlabReader(ContextReader):
            def describe_slab(self, sinks, source, sector='vsv', primary=None, **selection):
                metadata = self.describe_block(sinks[0], source, sector, primary, **selection)
                return replace(metadata, axes=('coordinate_batch', *metadata.axes),
                    shape=(len(sinks), *metadata.shape),
                    selection_identity=repr((primary, tuple(sinks), source)))

            def read_slab(self, sinks, source, sector='vsv', primary=None, **selection):
                return np.stack([self.block(sink, source, sector, primary, **selection) for sink in sinks])

        task, original, _, vertices = fixture()
        reader = SlabReader(original.raw, TimeSelectionRecord(128, (0, 1, 127)))
        actual = calculate(task, time_axes={'source': range(3), 'sink': range(3)},
                           **self.options(reader, vertices)).value
        expected = []
        for source in range(3):
            for sink in range(3):
                a = reader.context_readers[source].block(sink, source)[:, :, 0, 0]
                b = reader.context_readers[source].block(source, sink)[:, :, 0, 0]
                expected.append(-np.einsum('ab,ba->', a, b))
        np.testing.assert_allclose(np.asarray(actual).reshape(-1), expected, rtol=2e-13, atol=2e-13)

    def test_invalid_or_missing_context_roles_fail_before_numeric_reads(self):
        for axes in (('absent',), ('source', 'source'), ['source']):
            task, reader, _, vertices = fixture()
            reader.read_context_axes = axes
            with self.assertRaisesRegex(ValueError, 'read_context|read context'):
                calculate(task, time_axes=dict(source=0, sink=1), execute=False,
                          **self.options(reader, vertices))
            self.assertEqual(reader.reads, [])
