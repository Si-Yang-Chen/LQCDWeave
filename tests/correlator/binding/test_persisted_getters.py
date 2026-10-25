import unittest

from lqcdweave.contraction.binding.api import bind_meson_getters


class Getter:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def get(self, *times):
        self.calls.append((self.name, times))
        return self.name, times


class PersistedMesonGetterBindingTests(unittest.TestCase):
    def test_binding_is_lazy_and_routes_saved_handle_and_times(self):
        calls = []
        vertices = [Getter("v0", calls)]
        propagators = [Getter("p0", calls)]
        loader = bind_meson_getters(23, vertices=vertices, propagators=propagators)

        self.assertEqual(calls, [])
        self.assertEqual(
            loader(23, "vertex:0:11", {
                "kind": "vertex", "handle": 0, "method": "get", "times": [11],
            }),
            ("v0", (11,)),
        )
        self.assertEqual(
            loader(23, "propagator:0:11:17", {
                "kind": "propagator", "handle": 0, "method": "get", "times": [11, 17],
            }),
            ("p0", (11, 17)),
        )
        self.assertEqual(calls, [("v0", (11,)), ("p0", (11, 17))])

    def test_cfg_mismatch_fails_before_getter_read(self):
        calls = []
        loader = bind_meson_getters(23, vertices=[Getter("v0", calls)], propagators=[])

        with self.assertRaisesRegex(ValueError, "another configuration"):
            loader(24, "vertex:0:11", {
                "kind": "vertex", "handle": 0, "method": "get", "times": [11],
            })
        self.assertEqual(calls, [])

    def test_unsupported_routes_and_invalid_indices_fail_without_reads(self):
        calls = []
        loader = bind_meson_getters(23, vertices=[Getter("v0", calls)], propagators=[])
        base = {"kind": "vertex", "handle": 0, "method": "get", "times": [11]}

        with self.assertRaisesRegex(ValueError, "getter method"):
            loader(23, "k", dict(base, method="load"))
        with self.assertRaisesRegex(ValueError, "input kind"):
            loader(23, "k", dict(base, kind="other"))
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            loader(23, "k", dict(base, handle=-1))
        with self.assertRaisesRegex(ValueError, "integer labels"):
            loader(23, "k", dict(base, times=[True]))
        with self.assertRaisesRegex(IndexError, "no vertex getter handle"):
            loader(23, "k", dict(base, handle=1))
        self.assertEqual(calls, [])

    def test_getter_exceptions_propagate_and_b_does_not_own_handles(self):
        closed = []

        class Broken:
            def get(self, *_times):
                raise RuntimeError("getter sentinel")

            def close(self):
                closed.append(True)

        handles = [Broken()]
        loader = bind_meson_getters(23, vertices=handles, propagators=[])
        with self.assertRaisesRegex(RuntimeError, "getter sentinel"):
            loader(23, "vertex:0:11", {
                "kind": "vertex", "handle": 0, "method": "get", "times": [11],
            })
        self.assertEqual(closed, [])


if __name__ == "__main__":
    unittest.main()