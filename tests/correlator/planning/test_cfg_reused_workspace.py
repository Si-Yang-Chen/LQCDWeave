"""A reused Action does not execute its saved temporary workspace."""
from dataclasses import replace
import unittest

from correlator.planning import test_cfg_plan as fixtures
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.planning import cfg_plan


class CfgReusedWorkspaceTests(unittest.TestCase):
    def test_repeated_point_omits_nonexecuted_workspace_in_both_resource_domains(self):
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                selected, preps, specs = fixtures.CfgPlanTests.dense_fixture(backend)
                first = next(i for i, action in enumerate(selected.state.actions) if action.kind == 'load')
                domain = 'host' if backend == 'numpy' else 'device'
                points = []
                for point in selected.memory:
                    if point.action_index == first:
                        workspace = f'workspace_{domain}_bytes'
                        point = replace(point, **{workspace: 1 << 16})
                    points.append(point)
                selected = replace(selected, memory=tuple(points),
                    **{f'peak_{domain}_bytes': max(getattr(p, f'{domain}_bytes') for p in points)})
                coordinates, _ = describe_coordinates(time_pairs=[{0:0, 1:0}, {0:0, 1:0}])
                normal = cfg_plan.build_cfg_plan(selected, None, coordinates.points, {}, preps, specs)
                self.assertTrue(all(normal.instances[1].reused.array()))
                audited = cfg_plan.audit_cfg_plan(normal)
                self.assertEqual(audited[f'peak_{domain}_bytes'], getattr(normal, f'peak_{domain}_bytes'))
                debug = cfg_plan.build_cfg_plan(selected, None, coordinates.points, {}, preps, specs, debug=True)
                self.assertEqual(replace(normal, statistics={}), replace(debug, statistics={}))


if __name__ == '__main__':
    unittest.main()
