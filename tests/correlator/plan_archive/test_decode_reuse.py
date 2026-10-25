"""A complete cfg plan stores one immutable scene template for all points."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.storage.contraction_plans import api as archive_api
from lqcdweave.contraction.correlator import calc_diagram_eval, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation


class DecodeReuseTests(unittest.TestCase):
    def test_one_template_decodes_once_for_four_distinct_time_pairs(self):
        rng = np.random.default_rng(418)
        component = ElementalComponent(0)
        operator = Operator.from_terms("local", [OperatorTerm(0, component, 0)])
        elemental = ArrayElemental(
            {"v2v": rng.normal(size=(1, 1, 3, 1, 1))
             + 1j * rng.normal(size=(1, 1, 3, 1, 1))},
            (component,), phase_convention="anchor", cfg="cfg",
            data_identity="decode-reuse-elemental",
        )
        propagator = ArrayPropagator(
            {"vsv": rng.normal(size=(3, 3, 4, 4, 1, 1))
             + 1j * rng.normal(size=(3, 3, 4, 4, 1, 1))},
            time_convention="absolute", cfg="cfg",
            data_identity="decode-reuse-propagator",
        )
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]),
                          [0, 1], ["a", "b"], [None, "q"])
        options = dict(
            propagator_map={"q": propagator},
            vertex_map={name: Meson(elemental, operator, phase_convention="anchor")
                        for name in ("a", "b")},
            configuration="cfg", usedNe=1,
            target_hardware=TargetHardwareMemory(host_available_bytes=1 << 20),
            memory_limit=1 << 20, optimize="none",
            execute=False,
        )
        pairs = ({0: 2, 1: 0}, {0: 2, 1: 1}, {0: 0, 1: 1}, {0: 1, 1: 2})
        scene = archive_api.contract_api.SceneStoragePlan
        initializer = scene.__init__
        decoded_templates = []

        def track(instance, *args, **kwargs):
            decoded_templates.append(instance)
            initializer(instance, *args, **kwargs)

        with TemporaryDirectory() as directory:
            archive = Path(directory) / "all-pairs.json.gz"
            selected = calc_diagram_eval(diagram,time_axes={0:1,1:0},**options)
            coordinates, _ = describe_coordinates(time_pairs=pairs)
            original = complete_calculation(selected,coordinates=coordinates,
                readers=selected.inputs.loader.input_readers)
            save_calculation_plan(archive, original)
            with patch.object(scene, "__init__", track):
                restored = calc_diagram_eval(
                    plan_archive=archive,configuration="cfg",
                    propagator_map={"q":propagator},vertex_map=elemental,
                    memory_limit=1<<20,execute=False,
                )
            self.assertEqual(len(decoded_templates),1)
            self.assertEqual(len(restored.plan.templates),1)
            self.assertEqual(len(restored.plan.instances),len(pairs))
            actual = restored.run().value
            reference = calc_diagram_eval(
                diagram, time_pairs=pairs, **dict(options, execute=True),
            ).value
            np.testing.assert_allclose(actual, reference)


if __name__ == "__main__":
    unittest.main()
