"""Independent CPU checks adapted from old test_temporal_current_elemental.py."""

import unittest

import numpy as np

from lqcdweave.contracts.current import build_current_raw_contract, validate_current_raw_contract
from lqcdweave.production.elemental import ElementalGenerator
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


class _Handle:
    def __init__(self, data):
        self.data = data

    def load(self, key):
        return self.data


class DirectedCurrentProductionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2709)
        self.lattice = (2, 2, 1, 3)
        lx, ly, lz, lt = self.lattice
        self.links = (rng.normal(size=(lt, lz, ly, lx, 4, 3, 3))
                      + 1j * rng.normal(size=(lt, lz, ly, lx, 4, 3, 3)))
        self.vectors = (rng.normal(size=(lt, 2, lz, ly, lx, 3))
                        + 1j * rng.normal(size=(lt, 2, lz, ly, lx, 3)))
        self.momenta = ((0, 0, 0), (1, -1, 0))
        self.generator = ElementalGenerator(
            self.lattice, _Handle(self.links), _Handle(self.vectors),
            momentum_list=self.momenta, array_module=np,
            paths=[(d,) for d in range(8)],path_version="v2",phase_convention="midpoint",
        ).load("cfg")

    def _reference(self, direction, time, momentum):
        lx, ly, lz, lt = self.lattice
        left = self.vectors[time]
        gauge_axis = direction % 3
        if direction == 6:
            if time == lt - 1 and self.boundary == "open":
                return np.zeros((2, 2), complex)
            right = self.vectors[(time + 1) % lt]
            link = self.links[time, ..., 3, :, :]
            shift = (0, 0, 0)
        elif direction == 7:
            if time == 0 and self.boundary == "open":
                return np.zeros((2, 2), complex)
            previous = (time - 1) % lt
            right = self.vectors[previous]
            link = self.links[previous, ..., 3, :, :].conj().swapaxes(-1, -2)
            shift = (0, 0, 0)
        elif direction < 3:
            right = np.roll(left, -1, axis=3 - gauge_axis)
            link = self.links[time, ..., gauge_axis, :, :]
            shift = tuple(int(i == gauge_axis) for i in range(3))
        else:
            right = np.roll(left, 1, axis=3 - gauge_axis)
            link = np.roll(self.links[time, ..., gauge_axis, :, :], 1,
                           axis=2 - gauge_axis).conj().swapaxes(-1, -2)
            shift = tuple(-int(i == gauge_axis) for i in range(3))
        z, y, x = np.indices((lz, ly, lx))
        phase = np.exp(2j * np.pi * sum(
            p * (coord + delta / 2) / extent
            for p, coord, delta, extent in zip(
                momentum, (x, y, z), shift, (lx, ly, lz))))
        return np.einsum("zyx,ezyxa,zyxac,fzyxc->ef",
                         phase, left.conj(), link, right)

    def test_all_directions_periodic_and_open(self):
        for self.boundary in ("periodic", "open"):
            self.generator.path_boundary = self.boundary
            values = np.stack([self.generator.calc_path_blocks(t)["v2v"].copy() for t in range(3)], axis=1)
            result = {"v2v": values, "contract": build_current_raw_contract(
                {"v2v": values},boundary=self.boundary,available_ne=2,used_ne=2,momentum_count=2)}
            values = result["v2v"]
            self.assertEqual(values.shape, (8, 3, 2, 2, 2))
            self.assertEqual(result["contract"]["boundary"], self.boundary)
            validate_current_raw_contract({"v2v": values}, result["contract"],
                                          require_temporal=True)
            for direction in range(8):
                for time in range(3):
                    for index, momentum in enumerate(self.momenta):
                        np.testing.assert_allclose(
                            values[direction, time, index],
                            self._reference(direction, time, momentum),
                            rtol=1e-12, atol=1e-12,
                        )

    def test_loaded_inputs_and_contract_are_required(self):
        generator = ElementalGenerator(
            self.lattice, _Handle(self.links), _Handle(self.vectors),
        )
        with self.assertRaisesRegex(ValueError, "load inputs"):
            generator.calc_path_blocks(0)
        values = np.stack([self.generator.calc_path_blocks(t)["v2v"].copy() for t in range(3)], axis=1)
        raw = {"v2v": values, "contract": build_current_raw_contract(
            {"v2v": values},boundary="periodic",available_ne=2,used_ne=2,momentum_count=2)}
        forged = dict(raw["contract"], boundary="open")
        with self.assertRaisesRegex(ValueError, "cache identity"):
            validate_current_raw_contract({"v2v": raw["v2v"]}, forged,
                                          require_temporal=True)



if __name__ == "__main__":
    unittest.main()
