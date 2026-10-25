"""Loader-local binding of complete frozen cfg preparation templates."""
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
import gc
import importlib
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, CfgTemplate, SourcePreparation, TensorPreparation, TensorSpec,
)
from lqcdweave.contraction.execution.cfg_runtime import _LocalLoader
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.contraction.execution.preparation_kernel import FusedSpinPreparation


def _spin(spin, *, raw="raw", conjugate=False, permutation=(), profile=(), configuration=None):
    constants = tuple(ArrayConstant.from_array(value) for value in (spin, *profile))
    recipe = TensorPreparation("spin_spatial", (raw,), constants,
        conjugate=conjugate, permutation=permutation,
        configuration=configuration if profile else None)
    return SourcePreparation((raw,), recipe=recipe)


def _template(closure, specs, *, keys=None):
    return SimpleNamespace(
        keys=tuple(keys if keys is not None else closure),
        preparation_closure=MappingProxyType(dict(closure)),
        preparation_specs=MappingProxyType(dict(specs)),
    )


def _backend():
    class Backend:
        __name__ = "cupy"

        @staticmethod
        def asarray(value):
            return np.array(value, copy=True)

        @staticmethod
        def ElementwiseKernel(*args, **kwargs):
            raise AssertionError("the protocol test replaces numeric materialization")

    return Backend()


def _global_tables(closure, specs, suffix, raw_ids):
    """Create one current instance's global IDs and recipe raw IDs."""
    role_ids = {key: f"{key}-{suffix}" for key in closure}
    preparations, global_specs = {}, {}
    for key, prep in closure.items():
        recipe = prep.recipe
        if recipe.kind == "spin_spatial":
            raw_id = raw_ids[key]
            recipe = replace(recipe, inputs=(raw_id,))
            blocks = (raw_id,)
        elif recipe.kind == "stack":
            recipe = replace(recipe, inputs=tuple(role_ids[item] for item in recipe.inputs))
            blocks = tuple(dict.fromkeys(raw_ids.values()))
        else:
            recipe = replace(recipe)
            blocks = prep.blocks
        preparations[role_ids[key]] = replace(prep, blocks=blocks, recipe=recipe)
        if key in specs:
            global_specs[role_ids[key]] = specs[key]
    return role_ids, preparations, global_specs


def _profile_reference(spin, raw, left, right, weights, *, conjugate, permutation):
    projected = np.empty(raw.shape, dtype=np.complex128)
    d1 = np.result_type(weights.dtype, left.dtype)
    d2 = np.result_type(d1, raw.dtype)
    d3 = np.result_type(d2, right.dtype)
    for row, col in np.ndindex(raw.shape):
        total = np.complex128(0)
        for p in range(len(weights)):
            first = d1.type(d1.type(weights[p]) * d1.type(left[p, row].conjugate()))
            second = d2.type(d2.type(first) * d2.type(raw[row, col]))
            third = d3.type(d3.type(second) * d3.type(right[p, col]))
            total = np.complex128(total + np.complex128(third))
        projected[row, col] = total
    value = projected.conj() if conjugate else projected
    if permutation:
        value = value.transpose(permutation)
    return np.einsum("ab,...->ab...", spin, value, optimize=False)


class CfgTemplatePreparationBindingTests(unittest.TestCase):
    def _simple_tree(self, spin=None, *, dtype=np.complex128, configuration="cfg-a", profile=True):
        spin = np.eye(4, dtype=dtype) if spin is None else np.asarray(spin, dtype=dtype)
        profile_values = ((np.ones((2, 3), dtype=np.complex64),
                           np.ones((2, 2), dtype=np.complex64),
                           np.ones((2,), dtype=np.complex64)) if profile else ())
        leaf = _spin(spin, raw="raw-leaf", conjugate=True, permutation=(1, 0),
                     profile=profile_values, configuration=configuration)
        closure = {"leaf": leaf}
        specs = {"leaf": TensorSpec((4, 4, 2, 3), np.complex128 if profile else dtype)}
        return closure, specs

    def test_same_template_rebinds_current_gids_and_raw_arrays_without_retaining_them(self):
        closure = {
            "a": _spin(np.eye(4, dtype=np.float64), raw="raw-a"),
            "b": _spin(np.full((4, 4), 2.0), raw="raw-b"),
            "inner": SourcePreparation(("raw-a", "raw-b"), recipe=TensorPreparation(
                "stack", ("a", "b"))),
            "outer": SourcePreparation(("raw-a", "raw-b"), recipe=TensorPreparation(
                "stack", ("inner", "inner"))),
        }
        specs = {"a": TensorSpec((4, 4, 2, 3), np.float64),
                 "b": TensorSpec((4, 4, 2, 3), np.float64),
                 "inner": TensorSpec((2, 4, 4, 2, 3), np.float64),
                 "outer": TensorSpec((2, 2, 4, 4, 2, 3), np.float64)}
        template = _template(closure, specs)
        tables, bindings, raw_sets = {}, [], []
        for point in range(2):
            raw_ids = {"a": f"raw-a-{point}", "b": f"raw-b-{point}"}
            role_ids, preps, global_specs = _global_tables(closure, specs, point, raw_ids)
            tables.update(preps)
            bindings.append(role_ids)
            raw_sets.append({raw_ids["a"]: np.arange(6, dtype=np.float64).reshape(2, 3) + point,
                             raw_ids["b"]: np.arange(6, dtype=np.float64).reshape(2, 3) + 10 + point})
            if point == 0:
                all_specs = dict(global_specs)
            else:
                all_specs.update(global_specs)
        loader = bind_prepared_loader(None, MappingProxyType(tables),
                                      MappingProxyType(all_specs), backend=_backend())
        bound = loader.bind_template_preparations(template)
        self.assertIsNotNone(bound)
        self.assertIs(loader.bind_template_preparations(template), bound)
        self.assertEqual(len(bound.roles["outer"][1]), 4)

        seen = []
        def protocol_materialize(program, blocks, destination=None, *, inputs=None):
            seen.append((program, tuple(id(value) for value in inputs)))
            return np.array(inputs[0], copy=True)

        input_refs = []
        with patch.object(FusedSpinPreparation, "materialize", protocol_materialize):
            results = []
            for role_ids, raw_values in zip(bindings, raw_sets, strict=True):
                input_refs.extend(weakref.ref(value) for value in raw_values.values())
                local = _LocalLoader(loader, role_ids, bound)
                results.append(local.materialize("outer", None, raw_values))
        np.testing.assert_array_equal(results[0], raw_sets[0]["raw-a-0"])
        np.testing.assert_array_equal(results[1], raw_sets[1]["raw-a-1"])
        self.assertIs(seen[0][0], seen[1][0])
        self.assertNotEqual(seen[0][1], seen[1][1])
        self.assertEqual(len(loader._template_preparation_bindings), 1)
        self.assertEqual(loader._fused_bindings, {})
        del results, raw_sets, raw_values
        gc.collect()
        self.assertTrue(all(reference() is None for reference in input_refs))

    def test_complete_descriptions_separate_same_shape_constants_config_and_dtype(self):
        first_closure, first_specs = self._simple_tree()
        second_closure, second_specs = self._simple_tree(
            np.full((4, 4), 3.0), configuration="cfg-b")
        third_closure, third_specs = self._simple_tree(
            np.eye(4, dtype=np.complex64), dtype=np.complex64, configuration="cfg-c",
            profile=False)
        loader = bind_prepared_loader(None, MappingProxyType({}), MappingProxyType({}),
                                      backend=_backend())
        first = loader.bind_template_preparations(_template(first_closure, first_specs))
        second = loader.bind_template_preparations(_template(second_closure, second_specs))
        third = loader.bind_template_preparations(_template(third_closure, third_specs))
        first_program = first.roles["leaf"][0]
        second_program = second.roles["leaf"][0]
        third_program = third.roles["leaf"][0]
        self.assertIsNot(first_program, second_program)
        self.assertIsNot(first_program, third_program)
        self.assertIsNot(first_program.leaves[0].constants[0], second_program.leaves[0].constants[0])
        self.assertEqual(first_program.leaves[0].recipe.configuration, "cfg-a")
        self.assertEqual(second_program.leaves[0].recipe.configuration, "cfg-b")
        self.assertEqual(third_program.spec.dtype, np.dtype(np.complex64))

    def test_frozen_input_lookup_does_not_read_recipes_during_replay(self):
        closure, specs = self._simple_tree(dtype=np.float64, profile=False)

        class RecipeProbe:
            def __init__(self, recipe):
                self.value, self.reads = recipe, 0

            @property
            def recipe(self):
                self.reads += 1
                if self.reads > 1:
                    raise AssertionError("replay reread a frozen recipe")
                return self.value

        records = {str(gid): RecipeProbe(replace(
            closure["leaf"].recipe, inputs=(f"raw-{gid}",))) for gid in (42, 43)}
        loader = bind_prepared_loader(None, MappingProxyType(records),
            MappingProxyType({str(gid): specs["leaf"] for gid in (42, 43)}),
            backend=_backend())
        self.assertIsNone(loader._template_raw_inputs)
        binding = loader.bind_template_preparations(_template(closure, specs))

        def materialize(program, blocks, destination=None, *, inputs=None):
            return np.array(inputs[0], copy=True)

        references = []
        with patch.object(FusedSpinPreparation, "materialize", materialize):
            for gid in (42, 43, 42):
                raw = np.full((3, 2), gid, dtype=np.float64)
                references.append(weakref.ref(raw))
                actual = loader.materialize_template_preparation(
                    binding, "leaf", {"leaf": gid}, {f"raw-{gid}": raw})
                np.testing.assert_array_equal(actual, raw)
        self.assertEqual([record.reads for record in records.values()], [1, 1])
        del raw
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_native_template_identity_hit_does_not_rescan_frozen_maps(self):
        closure, specs = self._simple_tree()
        template = CfgTemplate(None, tuple(closure), None,
            MappingProxyType(closure), MappingProxyType(specs))
        loader = bind_prepared_loader(None, MappingProxyType({}), MappingProxyType({}),
                                      backend=_backend())
        binding = loader.bind_template_preparations(template)

        class _Unscannable:
            def items(self):
                raise AssertionError("native template cache hit rescanned its closure")

        original = template.preparation_closure
        object.__setattr__(template, "preparation_closure", _Unscannable())
        try:
            # Widen only this module's proxy check so an accidental descriptor
            # walk reaches the exploding mapping; the native identity hit must
            # return before it reads the already-frozen fields.
            with patch("lqcdweave.contraction.execution.preparation.MappingProxyType", object):
                self.assertIs(loader.bind_template_preparations(template), binding)
        finally:
            object.__setattr__(template, "preparation_closure", original)

    def test_protocol_template_replacement_rechecks_full_descriptor(self):
        closure, specs = self._simple_tree()
        template = _template(closure, specs)
        loader = bind_prepared_loader(None, MappingProxyType({}), MappingProxyType({}),
                                      backend=_backend())
        first = loader.bind_template_preparations(template)
        replacement, replacement_specs = self._simple_tree(
            np.full((4, 4), 4.0), configuration="cfg-replaced")
        template.preparation_closure = MappingProxyType(replacement)
        template.preparation_specs = MappingProxyType(replacement_specs)
        second = loader.bind_template_preparations(template)
        self.assertIsNot(first, second)
        self.assertEqual(second.roles["leaf"][0].leaves[0].recipe.configuration,
                         "cfg-replaced")

    def test_cache_is_bounded_to_sixteen_strong_template_identities(self):
        loader = bind_prepared_loader(None, MappingProxyType({}), MappingProxyType({}),
                                      backend=_backend())
        templates = [_template({}, {}, keys=(f"template-{index}",)) for index in range(17)]
        for template in templates:
            self.assertIsNotNone(loader.bind_template_preparations(template))
        self.assertEqual(len(loader._template_preparation_bindings), 16)
        self.assertIs(templates[-1], loader._template_preparation_bindings[id(templates[-1])].template)

    def test_cpu_old_template_and_mutable_tables_keep_original_materialize_path(self):
        closure, specs = self._simple_tree(dtype=np.float64, profile=False)
        global_id = "leaf-current"
        raw = np.arange(6, dtype=np.float64).reshape(3, 2)
        current = replace(closure["leaf"], recipe=replace(
            closure["leaf"].recipe, inputs=("raw-current",)))
        frozen = MappingProxyType({global_id: current})
        frozen_specs = MappingProxyType({global_id: specs["leaf"]})
        loader = bind_prepared_loader(None, frozen, frozen_specs, backend=np)
        full_template = _template(closure, specs)
        self.assertIsNone(loader.bind_template_preparations(full_template))
        local = _LocalLoader(loader, {"leaf": global_id}, None)
        actual = local.materialize("leaf", None, {"raw-current": raw})
        expected = np.einsum("ab,...->ab...", np.eye(4), raw.T.conj(), optimize=False)
        np.testing.assert_array_equal(actual, expected)

        mutable = bind_prepared_loader(None, {global_id: current}, {global_id: specs["leaf"]}, backend=np)
        self.assertIsNone(mutable.bind_template_preparations(full_template))
        mutable_local = _LocalLoader(mutable, {"leaf": global_id}, None)
        np.testing.assert_array_equal(
            mutable_local.materialize("leaf", None, {"raw-current": raw}), expected)

        old_template = SimpleNamespace(keys=("leaf",))
        old_binding = loader.bind_template_preparations(old_template)
        self.assertIsNone(old_binding)
        old_local = _LocalLoader(loader, {"leaf": global_id}, old_binding)
        np.testing.assert_array_equal(
            old_local.materialize("leaf", None, {"raw-current": raw}), expected)

    def test_loader_and_backend_scopes_do_not_share_bound_programs_or_constants(self):
        closure, specs = self._simple_tree()
        template = _template(closure, specs)
        first = bind_prepared_loader(None, MappingProxyType({}), MappingProxyType({}), backend=_backend())
        second = bind_prepared_loader(None, MappingProxyType({}), MappingProxyType({}), backend=_backend())
        first_program = first.bind_template_preparations(template).roles["leaf"][0]
        second_program = second.bind_template_preparations(template).roles["leaf"][0]
        self.assertIsNot(first_program, second_program)
        self.assertIsNot(first_program.leaves[0].constants[0], second_program.leaves[0].constants[0])

    def test_debug_uses_old_materializer_and_other_scope_rejects_binding(self):
        closure, specs = self._simple_tree(dtype=np.float64, profile=False)
        raw_ids = {"leaf": "raw-one"}
        role_ids, global_preps, global_specs = _global_tables(
            closure, specs, "one", raw_ids)
        template = _template(closure, specs)
        global_preps, global_specs = (MappingProxyType(global_preps),
                                      MappingProxyType(global_specs))
        first = bind_prepared_loader(None, global_preps, global_specs, backend=_backend())
        second = bind_prepared_loader(None, global_preps, global_specs, backend=_backend())
        binding = first.bind_template_preparations(template)
        blocks = {"raw-one": np.arange(6, dtype=np.float64).reshape(3, 2)}

        with self.assertRaisesRegex(ValueError, "another loader scope"):
            second.materialize_template_preparation(binding, "leaf", role_ids, blocks)

        calls = []
        def fake_materialize(program, raw_blocks, destination=None, *, inputs=None):
            calls.append(program)
            return np.zeros(program.spec.shape, dtype=program.spec.dtype)

        local = _LocalLoader(first, role_ids, binding)
        with patch.object(FusedSpinPreparation, "materialize", fake_materialize):
            with execution_debug(True):
                self.assertIsNone(first.materialize_template_preparation(
                    binding, "leaf", role_ids, blocks))
                actual = local.materialize("leaf", None, blocks)
        self.assertEqual(actual.shape, specs["leaf"].shape)
        self.assertEqual(len(calls), 1)
        self.assertIsNot(calls[0], binding.roles["leaf"][0])

        first_ref = weakref.ref(first)
        del local, first
        gc.collect()
        self.assertIsNone(first_ref())

    def test_missing_and_cyclic_template_members_fail_on_first_bind(self):
        spin, spec = self._simple_tree()
        missing = {"stack": SourcePreparation((), recipe=TensorPreparation("stack", ("absent",)))}
        loader = bind_prepared_loader(None, MappingProxyType({}), MappingProxyType({}), backend=_backend())
        with self.assertRaises(KeyError):
            loader.bind_template_preparations(_template(missing, {"stack": TensorSpec((1,), np.float64)}))
        cyclic = {"cycle": SourcePreparation((), recipe=TensorPreparation("stack", ("cycle",)))}
        with self.assertRaisesRegex(ValueError, "cyclic"):
            loader.bind_template_preparations(_template(cyclic, {"cycle": TensorSpec((1,), np.float64)}))


class CuPyCfgTemplatePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module("cupy")
            if cls.cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no device")
        except Exception as exc:
            raise unittest.SkipTest(f"CuPy device unavailable: {exc}")

    def test_nested_profile_dagger_casts_match_reference_and_debug_for_two_gid_sets(self):
        cp = self.cp
        rng = np.random.default_rng(8703)
        raw_a, raw_b = [], []
        leaf_values = {}
        for role, dtype in (("a", np.complex64), ("b", np.complex128)):
            spin = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))).astype(dtype)
            raw = (rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))).astype(np.complex64)
            left = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))).astype(np.complex64)
            right = (rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))).astype(np.complex128)
            weights = (rng.normal(size=3) + 1j * rng.normal(size=3)).astype(np.complex64)
            leaf_values[role] = (spin, raw, left, right, weights)
            (raw_a if role == "a" else raw_b).append(raw)

        closure = {
            role: _spin(values[0], raw=f"raw-{role}", conjugate=True, permutation=(1, 0),
                        profile=values[2:], configuration=f"cfg-{role}")
            for role, values in leaf_values.items()
        }
        closure["inner"] = SourcePreparation(("raw-a", "raw-b"), recipe=TensorPreparation(
            "stack", ("a", "b")))
        closure["outer"] = SourcePreparation(("raw-a", "raw-b"), recipe=TensorPreparation(
            "stack", ("inner", "inner")))
        specs = {"a": TensorSpec((4, 4, 2, 3), np.complex128),
                 "b": TensorSpec((4, 4, 2, 3), np.complex128),
                 "inner": TensorSpec((2, 4, 4, 2, 3), np.complex64),
                 "outer": TensorSpec((2, 2, 4, 4, 2, 3), np.complex128)}
        template = _template(closure, specs)

        global_preps, global_specs, instances = {}, {}, []
        for point in range(2):
            raw_ids = {"a": f"a-raw-{point}", "b": f"b-raw-{point}"}
            role_ids, preps, item_specs = _global_tables(closure, specs, point, raw_ids)
            global_preps.update(preps)
            global_specs.update(item_specs)
            instances.append((role_ids, raw_ids))
        loader = bind_prepared_loader(None, MappingProxyType(global_preps),
            MappingProxyType(global_specs), backend=cp)
        template_binding = loader.bind_template_preparations(template)
        self.assertIsNotNone(template_binding)
        outer_program = template_binding.roles["outer"][0]

        def reference(raw_by_role):
            a = leaf_values["a"]
            b = leaf_values["b"]
            result_a = _profile_reference(a[0], raw_by_role["a"], *a[2:],
                conjugate=True, permutation=(1, 0))
            result_b = _profile_reference(b[0], raw_by_role["b"], *b[2:],
                conjugate=True, permutation=(1, 0))
            inner = np.stack((result_a, result_b)).astype(np.complex64)
            return np.stack((inner, inner)).astype(np.complex128)

        outputs = []
        input_refs = []
        for point, (role_ids, raw_ids) in enumerate(instances):
            raw_host = {"a": leaf_values["a"][1] * (point + 1),
                        "b": leaf_values["b"][1] * (2 - point)}
            device_values = {raw_ids[role]: cp.asarray(value) for role, value in raw_host.items()}
            input_refs.extend(weakref.ref(value) for value in device_values.values())
            local = _LocalLoader(loader, role_ids, template_binding)
            output = local.materialize("outer", None, device_values)
            np.testing.assert_allclose(cp.asnumpy(output), reference(raw_host),
                                       rtol=8e-6, atol=8e-6)
            outputs.append(output)
            self.assertIs(template_binding.roles["outer"][0], outer_program)

        first_roles, first_raw_ids = instances[0]
        debug_raw_host = {"a": leaf_values["a"][1] * 3,
                          "b": leaf_values["b"][1] * 4}
        debug_blocks = {first_raw_ids[role]: cp.asarray(value)
                        for role, value in debug_raw_host.items()}
        debug_loader = _LocalLoader(loader, first_roles, template_binding)
        with execution_debug(True):
            debug_value = debug_loader.materialize("outer", None, debug_blocks)
        np.testing.assert_allclose(cp.asnumpy(debug_value), reference(debug_raw_host),
                                   rtol=8e-6, atol=8e-6)
        self.assertIsNot(outputs[0], outputs[1])
        self.assertNotEqual(outputs[0].data.ptr, outputs[1].data.ptr)
        self.assertEqual(len(loader._template_preparation_bindings), 1)
        del debug_blocks, device_values, outputs, debug_value
        gc.collect()
        self.assertTrue(all(reference() is None for reference in input_refs))


if __name__ == "__main__":
    unittest.main()
