"""Module tests split from test_laplace_backend.py; shared numerical fixtures stay unchanged."""

from fixtures.laplace_backend import (
    GaugeHandle,
    Mock,
    _LaplaceBackendFixture,
    np,
)


class LaplaceBackendTests(_LaplaceBackendFixture):
    def test_quda_native_and_file_loading_stays_in_adapter(self):
        adapter, native = self.adapter(), self.native()
        handle = Mock()
        handle.load.return_value = native
        links = adapter.load(handle, "cfg", self.sizes)
        self.assertFalse(np.shares_memory(links, native.lexico()))
        adapter._gauge = Mock(side_effect=AssertionError("unexpected repacking"))
        updated = adapter.stout_smear(links, self.sizes, 2, .1)
        adapter.stout_smear(updated, self.sizes, 1, .1)
        self.assertEqual(native.stoutSmear.call_count, 2)
        adapter.invalidate()
        adapter._gauge = Mock(return_value=(None, self.native()))
        adapter.stout_smear(updated, self.sizes, 1, .1)
        adapter._gauge.assert_called_once()
        file_handle = GaugeHandle(self.handle.data)
        file_handle.file = "explicit-gauge.lime"
        file_handle.format = "ildg"
        adapter.load_gauge = Mock(return_value=native)
        adapter.load(file_handle, "cfg", self.sizes)
        adapter.load_gauge.assert_called_once_with("explicit-gauge.lime", self.sizes)

    def test_quda_does_not_guess_qio_from_a_filename(self):
        adapter = self.adapter()
        adapter.load_gauge = Mock(side_effect=AssertionError("not QIO"))
        for format in ("npy", "binary", "qdp", None):
            with self.subTest(format=format):
                self.handle.file = "gauge.lime"
                self.handle.format = format
                links = adapter.load(self.handle, "cfg", self.sizes)
                np.testing.assert_array_equal(
                    links, self.handle.data.transpose(4, 0, 1, 2, 3, 5, 6))
                self.assertFalse(np.shares_memory(links, self.handle.data))
        adapter.load_gauge.assert_not_called()

    def test_quda_array_load_clears_cache_and_failure_cannot_reuse_it(self):
        adapter = self.adapter()
        handle = Mock()
        handle.load.return_value = self.native()
        adapter.load(handle, "cfg", self.sizes)
        links = adapter.load(self.handle, "array", self.sizes)
        self.assertIsNone(adapter._native_gauge)
        np.testing.assert_array_equal(links, self.handle.data.transpose(4, 0, 1, 2, 3, 5, 6))
        native = self.native()
        handle.load.return_value = native
        links = adapter.load(handle, "cfg", self.sizes)
        native.stoutSmear.side_effect = RuntimeError("native failure")
        with self.assertRaises(RuntimeError):
            adapter.stout_smear(links, self.sizes, 1, .1)
        self.assertIsNone(adapter._native_gauge)
        self.assertIsNone(adapter._native_links)
