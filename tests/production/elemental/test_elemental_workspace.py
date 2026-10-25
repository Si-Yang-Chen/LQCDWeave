"""Shared elemental workspace ownership and failure recovery."""

import unittest
from unittest.mock import patch
import numpy as np

from production.elemental.test_elemental import fixture, Handle
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator, BaryonElementalKernel as BaryonElementalGenerator
from lqcdweave.production.elemental import (
    ElementalWorkspace,
)


class WorkspaceTests(unittest.TestCase):
    def test_owned_links_and_geometric_phase(self):
        sizes, gauge, _, _ = fixture()
        original = gauge.copy()
        handle = Handle(gauge)
        workspace = ElementalWorkspace(sizes, handle).load("a")
        gauge.fill(0)
        np.testing.assert_array_equal(workspace.links, original.transpose(4,0,1,2,3,5,6))
        self.assertTrue(workspace.links.flags.c_contiguous)
        momentum, displacement = (1,-1,1), (1,0,-1)
        phase = workspace.phase(momentum, displacement)
        lx, ly, lz, _ = sizes
        z,y,x = np.indices((lz,ly,lx))
        expected = np.exp(2j*np.pi*((x+.5)/lx-y/ly+(z-.5)/lz))
        np.testing.assert_allclose(phase, expected)
        handle.array = original
        workspace.load("b")
        self.assertIs(phase, workspace.phase(momentum, displacement))
        buffer = workspace.buffer("tile", (2,3))
        self.assertIs(buffer, workspace.buffer("tile", (2,3)))
        self.assertEqual(workspace.buffer("tile", (1,3)).shape, (1,3))
        handle.array = original[0]
        with self.assertRaises(ValueError):
            workspace.load("invalid")
        with self.assertRaises(ValueError):
            _ = workspace.links

    def test_partial_transform_failure_requires_reload_in_both_producers(self):
        sizes, gauge, vectors, _ = fixture()
        def fail(links, *args, **kwargs):
            links.fill(0)
            raise RuntimeError("partial device operation")
        for cls in (ElementalGenerator, BaryonElementalGenerator):
            for method, kernel, args in (("stout_smear", "stout_smear", (1,.1)),
                                         ("project_SU3", "project_su3", ())):
                with self.subTest(producer=cls.__name__, method=method):
                    gen = cls(sizes, Handle(gauge), Handle(vectors)).load("cfg")
                    expected = next(iter(gen.calc_path_blocks(0).values())).copy()
                    with patch("lqcdweave.production.elemental.workspace."+kernel, fail):
                        with self.assertRaises(RuntimeError):
                            getattr(gen, method)(*args)
                    with self.assertRaises(ValueError):
                        next(iter(gen.calc_path_blocks(0).values()))
                    gen.load("cfg")
                    np.testing.assert_allclose(next(iter(gen.calc_path_blocks(0).values())), expected)

    def test_failed_vector_load_disables_gauge_operations(self):
        sizes, gauge, vectors, _ = fixture()
        for cls in (ElementalGenerator, BaryonElementalGenerator):
            with self.subTest(producer=cls.__name__):
                handle = Handle(vectors)
                gen = cls(sizes, Handle(gauge), handle).load("cfg")
                handle.array = vectors[0]
                with self.assertRaises(ValueError):
                    gen.load("invalid")
                with self.assertRaises(ValueError):
                    next(iter(gen.calc_path_blocks(0).values()))
                with self.assertRaises(ValueError):
                    gen.project_SU3()
                handle.array = vectors
                gen.load("cfg")
                self.assertTrue(np.all(np.isfinite(next(iter(gen.calc_path_blocks(0).values())))))
