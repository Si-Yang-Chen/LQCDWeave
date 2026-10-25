"""Private window transport preserves metadata semantics and current inputs."""
from dataclasses import replace
import gc
from io import BytesIO
import os
import pickle
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest
import weakref

import numpy as np
from correlator.execution import test_cfg_table_reuse as cfg_fixtures
from correlator.execution import test_cfg_runtime as runtime_fixtures
from lqcdweave.contracts.correlator.api import SourcePreparation, describe_coordinates
from lqcdweave.contracts.correlator.semantics import TensorDescription
from lqcdweave.contraction.execution.api import execute_cfg_plan, execute_cfg_windows
from lqcdweave.contraction.expansion import saved_inputs
from lqcdweave.contraction.planning.api import iter_cfg_plan_windows
from lqcdweave.contraction._cfg_plan_worker import WindowSender, WindowReceiver, send
from lqcdweave.contraction import _cfg_plan_worker as wire
from lqcdweave.contraction.correlator_cfg import CompleteCalculation, source_loader_factory


def transmit(sender, receiver, value):
    stream = BytesIO()
    sender.send(stream, value)
    size = stream.tell()
    stream.seek(0)
    return receiver.receive(stream), size


class WindowTransportTests(unittest.TestCase):
    def test_repeated_descriptors_resolve_once_and_preserve_aliases(self):
        windows, _ = cfg_fixtures.CfgTableReuseTests().windows(1)
        window = windows[0]
        template = window.templates[0]
        spec_key, spec = next(iter(template.preparation_specs.items()))
        preparation_key, preparation = next(iter(window.preparations.items()))
        values = (template, spec, preparation)*1000
        references = {id(template): ('cfg-template-v1', 0),
                      id(spec): ('cfg-spec-v1', (0, spec_key)),
                      id(preparation): ('cfg-preparation-v1', preparation_key)}
        reader = wire._TemplateUnpickler(BytesIO(wire.pack(values, template_refs=references)),
                                         (template,), window.preparations)
        resolve, calls = reader.persistent_load, []

        def counted(reference):
            calls.append(reference)
            return resolve(reference)

        reader.persistent_load = counted
        restored = reader.load()
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(value is values[index] for index, value in enumerate(restored)))

    def test_reference_scope_is_cleared_on_success_failure_and_nested_receive(self):
        preparation = SourcePreparation(('raw',))
        reference = ('cfg-preparation-v1', 'input')
        payload = wire.pack(preparation, template_refs={id(preparation): reference})
        outer = wire._TemplateUnpickler(BytesIO(payload), (), {'input': preparation})
        inner_value = SourcePreparation(('inner',))

        def nested(token):
            inner = wire._TemplateUnpickler(BytesIO(payload), (), {'input': inner_value})
            self.assertIs(inner.load(), inner_value)
            self.assertIs(wire._reference_resolver.get(), nested)
            return outer.preparations[token[1]]

        outer.persistent_load = nested
        self.assertIs(outer.load(), preparation)
        for malformed in (payload[:-1], wire.pack(preparation,
                template_refs={id(preparation): ('cfg-preparation-v1', 'missing')})):
            with self.subTest(payload=malformed), self.assertRaises((EOFError, pickle.UnpicklingError)):
                wire._TemplateUnpickler(BytesIO(malformed), (), {'input': preparation}).load()
        self.assertIsNone(wire._reference_resolver.get())
        with self.assertRaises(pickle.UnpicklingError):
            wire._restore_reference(reference)

    def test_concurrent_receivers_keep_their_own_previous_window(self):
        marker = SourcePreparation(('marker',))
        payload = wire.pack((marker,)*20,
                            template_refs={id(marker): ('cfg-preparation-v1', 'input')})
        barrier = Barrier(2)

        def decode(name):
            expected = SourcePreparation((name,))
            reader = wire._TemplateUnpickler(BytesIO(payload), (), {'input': expected})
            resolve = reader.persistent_load

            def waiting(reference):
                barrier.wait(timeout=10)
                return resolve(reference)

            reader.persistent_load = waiting
            values = reader.load()
            return all(value is expected for value in values), wire._reference_resolver.get()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(decode, name) for name in ('left', 'right')]
            self.assertEqual([future.result(timeout=15) for future in futures], [(True, None)]*2)

    def test_reused_reference_reducer_is_importable_in_a_separate_process(self):
        script = '''
import sys
from types import SimpleNamespace
from lqcdweave.contracts.correlator.api import SourcePreparation, TensorSpec
from lqcdweave.contraction._cfg_plan_worker import WindowSender
spec = TensorSpec((2, 2), 'float64')
template = SimpleNamespace(preparation_specs={'local': spec})
prep = SourcePreparation(('raw',))
window = SimpleNamespace(templates=(template,), preparations={'input': prep},
                         specs={str(i): spec for i in range(100)})
sender = WindowSender()
sender.send(sys.stdout.buffer, ('window', window))
sender.send(sys.stdout.buffer, ('window', window))
'''
        environment = dict(os.environ)
        source = str(Path(wire.__file__).resolve().parents[2])
        environment['PYTHONPATH'] = os.pathsep.join(filter(None,
            (source, environment.get('PYTHONPATH'))))
        result = subprocess.run([sys.executable, '-c', script], env=environment,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        receiver, stream = WindowReceiver(), BytesIO(result.stdout)
        first = receiver.receive(stream)[1]
        second = receiver.receive(stream)[1]
        self.assertIs(second.templates[0], first.templates[0])
        self.assertIs(second.preparations['input'], first.preparations['input'])
        self.assertTrue(all(spec is first.templates[0].preparation_specs['local']
                            for spec in second.specs.values()))
        self.assertEqual(stream.read(), b'')

    def test_reused_template_supplies_complete_fixed_specs(self):
        windows, _ = cfg_fixtures.CfgTableReuseTests().windows(2)
        sender, receiver = WindowSender(), WindowReceiver()
        previous = transmit(sender, receiver, ('window', windows[0]))[0][1]
        current = transmit(sender, receiver, ('window', windows[1]))[0][1]
        self.assertIs(current.templates[0], previous.templates[0])
        for key, original in windows[1].specs.items():
            matches = [local for local, spec in windows[1].templates[0].preparation_specs.items()
                       if spec is original]
            self.assertTrue(matches)
            self.assertEqual(current.specs[key].description, original.description)
            self.assertIs(current.specs[key], previous.templates[0].preparation_specs[matches[0]])

    def test_changed_scientific_spec_metadata_is_not_replaced(self):
        window = cfg_fixtures.CfgTableReuseTests().windows(1)[0][0]
        template = window.templates[0]
        local = next(iter(template.preparation_specs))
        original = template.preparation_specs[local]
        changed_spec = replace(original, description=TensorDescription(
            role='current-window', semantics={'scientific_identity': {'cfg': 'different'}}))
        # Numerical TensorSpec equality excludes description by design.
        self.assertEqual(changed_spec, original)
        changed_template = replace(template, preparation_specs=dict(
            template.preparation_specs, **{local: changed_spec}))
        changed = replace(window, templates=(changed_template,), specs={
            key: changed_spec if spec is original else spec for key, spec in window.specs.items()})
        self.assertEqual(changed_template, template)
        sender, receiver = WindowSender(), WindowReceiver()
        previous = transmit(sender, receiver, ('window', window))[0][1]
        current = transmit(sender, receiver, ('window', changed))[0][1]
        self.assertIsNot(current.templates[0], previous.templates[0])
        self.assertEqual(current.templates[0].preparation_specs[local].description,
                         changed_spec.description)
        for key, spec in changed.specs.items():
            self.assertEqual(current.specs[key].description, spec.description)

    def test_reused_template_keeps_current_constants_and_numeric_results(self):
        windows, expected = cfg_fixtures.CfgTableReuseTests().windows(3)
        sender, receiver = WindowSender(), WindowReceiver()
        decoded, sizes = [], []
        for window in windows:
            response, size = transmit(sender, receiver, ('window', window))
            restored = response[1]
            self.assertEqual(restored, window)
            decoded.append(restored)
            sizes.append(size)
        self.assertIs(decoded[0].templates[0], decoded[1].templates[0])
        key = next(iter(decoded[0].preparations))
        self.assertIsNot(decoded[0].preparations[key], decoded[1].preparations[key])
        self.assertLess(sizes[1], sizes[0])
        outputs = []
        factory = source_loader_factory(CompleteCalculation(decoded[0], options={'readers': {}}))
        execute_cfg_windows(iter(decoded), 'cfg', factory,
                            lambda cfg, index, values: outputs.append((index, values)))
        for index, values in outputs[:-1]:
            for actual, reference in zip(values, expected, strict=True):
                np.testing.assert_allclose(actual, reference*(index+1)**2, rtol=2e-13, atol=2e-13)

    def test_equal_preparation_copies_share_descriptors_and_keep_numeric_results(self):
        windows, expected = cfg_fixtures.CfgTableReuseTests().windows(3)
        first = windows[0]
        windows = [replace(window, preparations={key: replace(prep,
            recipe=replace(prep.recipe, constants=tuple(replace(value)
                for value in prep.recipe.constants)))
            for key, prep in first.preparations.items()}) for window in windows]
        sender, receiver = WindowSender(), WindowReceiver()
        decoded = [transmit(sender, receiver, ('window', window))[0][1] for window in windows]
        for key in first.preparations:
            self.assertIsNot(windows[0].preparations[key], windows[1].preparations[key])
            self.assertIs(decoded[0].preparations[key], decoded[1].preparations[key])
            self.assertIs(decoded[1].preparations[key], decoded[2].preparations[key])
        outputs = []
        factory = source_loader_factory(CompleteCalculation(decoded[0], options={'readers': {}}))
        execute_cfg_windows(iter(decoded), 'cfg', factory,
                            lambda cfg, index, values: outputs.append((index, values)))
        for index, values in outputs[:-1]:
            for actual, reference in zip(values, expected, strict=True):
                np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=2e-13)

    def test_reused_preparations_resolve_current_window_reads(self):
        plan, expanded, selected, _, _ = runtime_fixtures.CfgReplayTests().setup_plan()
        points, _ = describe_coordinates(time_pairs=[{0: 1, 1: 0}, {0: 0, 1: 1}])
        windows = list(iter_cfg_plan_windows(selected, plan.templates[0].output, points.points,
            saved_inputs.describe_input_reads(expanded.inputs), expanded.inputs.loader.preparations,
            expanded.inputs.loader.specs, window_size=1))
        sender, receiver = WindowSender(), WindowReceiver()
        decoded = [transmit(sender, receiver, ('window', window))[0][1] for window in windows]
        self.assertNotEqual(decoded[0].requests, decoded[1].requests)
        for key in decoded[0].preparations:
            self.assertIs(decoded[0].preparations[key], decoded[1].preparations[key])
        factory = source_loader_factory(CompleteCalculation(plan,
            options={'readers': expanded.inputs.loader.input_readers}))
        expected, actual = [], []
        execute_cfg_plan(plan, ('cfg',), factory,
                         lambda cfg, index, values: expected.append((index, values)))
        execute_cfg_windows(iter(decoded), 'cfg', factory,
                            lambda cfg, index, values: actual.append((index, values)))
        self.assertTrue(any(not np.array_equal(a, b) for a, b in
                            zip(expected[0][1], expected[1][1], strict=True)))
        self.assertEqual([index for index, _ in actual], [index for index, _ in expected])
        for (_, values), (_, references) in zip(actual[:-1], expected[:-1], strict=True):
            for value, reference in zip(values, references, strict=True):
                np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)

    def test_changed_dependencies_and_workspace_are_not_replaced(self):
        window = cfg_fixtures.CfgTableReuseTests().windows(1)[0][0]
        key = next(iter(window.preparations))
        preparation = window.preparations[key]
        changes = (replace(preparation, blocks=('current-raw',)),
                   replace(preparation, workspace_host_bytes=preparation.workspace_host_bytes+1),
                   replace(preparation, workspace_device_bytes=preparation.workspace_device_bytes+1))
        for changed in changes:
            with self.subTest(preparation=changed):
                sender, receiver = WindowSender(), WindowReceiver()
                first = transmit(sender, receiver, ('window', window))[0][1]
                current = replace(window, preparations=dict(window.preparations, **{key: changed}))
                second = transmit(sender, receiver, ('window', current))[0][1]
                self.assertEqual(second, current)
                self.assertIsNot(second.preparations[key], first.preparations[key])

    def test_shared_preparation_under_multiple_keys_preserves_aliases(self):
        windows, _ = cfg_fixtures.CfgTableReuseTests().windows(2)
        key = next(iter(windows[0].preparations))
        first = replace(windows[0], preparations={'left': windows[0].preparations[key],
                                                'right': windows[1].preparations[key]})
        shared = replace(first.preparations['left'])
        second = replace(windows[1], preparations={'left': shared, 'right': shared})
        sender, receiver = WindowSender(), WindowReceiver()
        previous = transmit(sender, receiver, ('window', first))[0][1]
        current = transmit(sender, receiver, ('window', second))[0][1]
        self.assertEqual(current, second)
        self.assertIs(current.preparations['left'], previous.preparations['left'])
        self.assertIs(current.preparations['left'], current.preparations['right'])

    def test_invalid_private_references_fail_at_receive_boundary(self):
        tokens = (('cfg-preparation-v1', 'missing'), ('cfg-preparation-v1', 1),
                  ('cfg-preparation-v2', 'known'), ('cfg-preparation-v1',),
                  ('cfg-template-v1', 0), ('cfg-template-v1', True),
                  ('cfg-spec-v1', (0, 'missing')), ('cfg-spec-v1', (True, 'known')),
                  ('cfg-spec-v1', (0, 1)), ('cfg-spec-v2', (0, 'known')),
                  ('cfg-spec-v1', (0,)), ('cfg-spec-v1', 'known'))
        for token in tokens:
            with self.subTest(reference=token):
                marker = object()
                stream = BytesIO()
                send(stream, marker, template_refs={id(marker): token})
                stream.seek(0)
                with self.assertRaises(pickle.UnpicklingError):
                    WindowReceiver().receive(stream)

    def test_receiver_releases_preparations_on_change_end_and_close(self):
        windows, _ = cfg_fixtures.CfgTableReuseTests().windows(3)
        sender, receiver = WindowSender(), WindowReceiver()
        first = transmit(sender, receiver, ('window', windows[0]))[0][1]
        key = next(iter(first.preparations))
        previous = weakref.ref(first.preparations[key])
        del first
        second = transmit(sender, receiver, ('window', windows[1]))[0][1]
        gc.collect()
        self.assertIsNone(previous())
        current = weakref.ref(second.preparations[key])
        del second
        self.assertEqual(transmit(sender, receiver, ('end', None))[0], ('end', None))
        gc.collect()
        self.assertIsNone(current())
        self.assertEqual(sender.preparations, {})
        third = transmit(sender, receiver, ('window', windows[2]))[0][1]
        current = weakref.ref(third.preparations[key])
        del third
        receiver.clear()
        gc.collect()
        self.assertIsNone(current())

    def test_changed_cost_is_not_replaced_by_stale_template(self):
        window = cfg_fixtures.CfgTableReuseTests().windows(1)[0][0]
        template = window.templates[0]
        original = template.plan.state
        actions = (replace(original.actions[0], seconds=original.actions[0].seconds+1),
                   *original.actions[1:])
        changed = replace(window, templates=(replace(template, plan=replace(template.plan,
            state=replace(original, actions=actions))),))
        sender, receiver = WindowSender(), WindowReceiver()
        first = transmit(sender, receiver, ('window', window))[0][1]
        second = transmit(sender, receiver, ('window', changed))[0][1]
        self.assertEqual(second, changed)
        self.assertIsNot(first.templates[0], second.templates[0])
        self.assertEqual(second.templates[0].plan.state.actions[0].seconds,
                         original.actions[0].seconds+1)

    def test_receiver_releases_previous_template_on_change_and_end(self):
        window = cfg_fixtures.CfgTableReuseTests().windows(1)[0][0]
        sender, receiver = WindowSender(), WindowReceiver()
        first = transmit(sender, receiver, ('window', window))[0][1]
        reference = weakref.ref(first.templates[0])
        del first
        template = window.templates[0]
        changed = replace(window, templates=(replace(template, plan=replace(template.plan,
            memory_limit=template.plan.memory_limit+1)),))
        second = transmit(sender, receiver, ('window', changed))[0][1]
        gc.collect()
        self.assertIsNone(reference())
        last = weakref.ref(second.templates[0])
        del second
        self.assertEqual(transmit(sender, receiver, ('end', None))[0], ('end', None))
        gc.collect()
        self.assertIsNone(last())


if __name__ == '__main__':
    unittest.main()
