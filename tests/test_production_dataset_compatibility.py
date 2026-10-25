"""Pinned accepted scientific records for production workflow migration."""

import json
import unittest

from lqcdweave.contracts.dataset import DatasetSpec, canonical_json
from lqcdweave.planning.laplace import (
    laplace_basis_document, laplace_eigenvalues_document,
)
from lqcdweave.production import wilson_reference

# Captured from accepted cdd6791 private record builders before relocation.
CASES = json.loads('[{"configuration":{"ensemble":"compatibility","configuration":"cfg","lattice":[2,1,1,3]},"times":[0,1,2],"ne":2,"tolerance":1e-11,"basis_json":"{\\"axes\\":[{\\"coordinates\\":[0,1,2],\\"name\\":\\"time\\"},{\\"coordinates\\":[0,1],\\"name\\":\\"mode\\"},{\\"coordinates\\":[0],\\"name\\":\\"z\\"},{\\"coordinates\\":[0],\\"name\\":\\"y\\"},{\\"coordinates\\":[0,1],\\"name\\":\\"x\\"},{\\"coordinates\\":[0,1,2],\\"name\\":\\"color\\"}],\\"bases\\":{},\\"configuration\\":{\\"configuration\\":\\"cfg\\",\\"ensemble\\":\\"compatibility\\",\\"lattice\\":[2,1,1,3]},\\"conventions\\":{\\"normalization\\":\\"orthonormal\\",\\"phase\\":\\"first_component_real\\",\\"spatial_boundary\\":\\"periodic\\"},\\"definition\\":{\\"gauge_id\\":\\"actual-gauge\\",\\"operator\\":\\"positive_spatial_laplacian\\",\\"solver_tolerance\\":1e-11},\\"dtype\\":\\"complex128\\",\\"kind\\":\\"laplace_basis\\",\\"parents\\":[\\"actual-gauge\\"],\\"schema\\":1}","eigenvalues_json":"{\\"axes\\":[{\\"coordinates\\":[0,1,2],\\"name\\":\\"time\\"},{\\"coordinates\\":[0,1],\\"name\\":\\"mode\\"}],\\"bases\\":{\\"mode\\":{\\"data_id\\":\\"actual-basis\\",\\"indices\\":[0,1]}},\\"configuration\\":{\\"configuration\\":\\"cfg\\",\\"ensemble\\":\\"compatibility\\",\\"lattice\\":[2,1,1,3]},\\"conventions\\":{},\\"definition\\":{\\"gauge_id\\":\\"actual-gauge\\",\\"operator\\":\\"positive_spatial_laplacian\\",\\"solver_tolerance\\":1e-11},\\"dtype\\":\\"float64\\",\\"kind\\":\\"laplace_eigenvalues\\",\\"parents\\":[\\"actual-basis\\",\\"actual-gauge\\"],\\"schema\\":1}"},{"configuration":{"ensemble":"compatibility","configuration":"cfg","lattice":[2,2,2,2]},"times":[1,0],"ne":3,"tolerance":1e-09,"basis_json":"{\\"axes\\":[{\\"coordinates\\":[1,0],\\"name\\":\\"time\\"},{\\"coordinates\\":[0,1,2],\\"name\\":\\"mode\\"},{\\"coordinates\\":[0,1],\\"name\\":\\"z\\"},{\\"coordinates\\":[0,1],\\"name\\":\\"y\\"},{\\"coordinates\\":[0,1],\\"name\\":\\"x\\"},{\\"coordinates\\":[0,1,2],\\"name\\":\\"color\\"}],\\"bases\\":{},\\"configuration\\":{\\"configuration\\":\\"cfg\\",\\"ensemble\\":\\"compatibility\\",\\"lattice\\":[2,2,2,2]},\\"conventions\\":{\\"normalization\\":\\"orthonormal\\",\\"phase\\":\\"first_component_real\\",\\"spatial_boundary\\":\\"periodic\\"},\\"definition\\":{\\"gauge_id\\":\\"actual-gauge\\",\\"operator\\":\\"positive_spatial_laplacian\\",\\"solver_tolerance\\":1e-09},\\"dtype\\":\\"complex128\\",\\"kind\\":\\"laplace_basis\\",\\"parents\\":[\\"actual-gauge\\"],\\"schema\\":1}","eigenvalues_json":"{\\"axes\\":[{\\"coordinates\\":[1,0],\\"name\\":\\"time\\"},{\\"coordinates\\":[0,1,2],\\"name\\":\\"mode\\"}],\\"bases\\":{\\"mode\\":{\\"data_id\\":\\"actual-basis\\",\\"indices\\":[0,1,2]}},\\"configuration\\":{\\"configuration\\":\\"cfg\\",\\"ensemble\\":\\"compatibility\\",\\"lattice\\":[2,2,2,2]},\\"conventions\\":{},\\"definition\\":{\\"gauge_id\\":\\"actual-gauge\\",\\"operator\\":\\"positive_spatial_laplacian\\",\\"solver_tolerance\\":1e-09},\\"dtype\\":\\"float64\\",\\"kind\\":\\"laplace_eigenvalues\\",\\"parents\\":[\\"actual-basis\\",\\"actual-gauge\\"],\\"schema\\":1}"},{"configuration":{"ensemble":"compatibility","configuration":"cfg","lattice":[3,1,1,1]},"times":[0],"ne":1,"tolerance":1e-10,"basis_json":"{\\"axes\\":[{\\"coordinates\\":[0],\\"name\\":\\"time\\"},{\\"coordinates\\":[0],\\"name\\":\\"mode\\"},{\\"coordinates\\":[0],\\"name\\":\\"z\\"},{\\"coordinates\\":[0],\\"name\\":\\"y\\"},{\\"coordinates\\":[0,1,2],\\"name\\":\\"x\\"},{\\"coordinates\\":[0,1,2],\\"name\\":\\"color\\"}],\\"bases\\":{},\\"configuration\\":{\\"configuration\\":\\"cfg\\",\\"ensemble\\":\\"compatibility\\",\\"lattice\\":[3,1,1,1]},\\"conventions\\":{\\"normalization\\":\\"orthonormal\\",\\"phase\\":\\"first_component_real\\",\\"spatial_boundary\\":\\"periodic\\"},\\"definition\\":{\\"gauge_id\\":\\"actual-gauge\\",\\"operator\\":\\"positive_spatial_laplacian\\",\\"solver_tolerance\\":1e-10},\\"dtype\\":\\"complex128\\",\\"kind\\":\\"laplace_basis\\",\\"parents\\":[\\"actual-gauge\\"],\\"schema\\":1}","eigenvalues_json":"{\\"axes\\":[{\\"coordinates\\":[0],\\"name\\":\\"time\\"},{\\"coordinates\\":[0],\\"name\\":\\"mode\\"}],\\"bases\\":{\\"mode\\":{\\"data_id\\":\\"actual-basis\\",\\"indices\\":[0]}},\\"configuration\\":{\\"configuration\\":\\"cfg\\",\\"ensemble\\":\\"compatibility\\",\\"lattice\\":[3,1,1,1]},\\"conventions\\":{},\\"definition\\":{\\"gauge_id\\":\\"actual-gauge\\",\\"operator\\":\\"positive_spatial_laplacian\\",\\"solver_tolerance\\":1e-10},\\"dtype\\":\\"float64\\",\\"kind\\":\\"laplace_eigenvalues\\",\\"parents\\":[\\"actual-basis\\",\\"actual-gauge\\"],\\"schema\\":1}"}]')


class ProductionDatasetCompatibilityTests(unittest.TestCase):
    def test_default_laplace_records_preserve_accepted_bytes(self):
        for case in CASES:
            with self.subTest(lattice=case['configuration']['lattice'],ne=case['ne']):
                args=(case['configuration'],case['times'],'actual-gauge',case['ne'],case['tolerance'])
                basis=laplace_basis_document(*args)
                values=laplace_eigenvalues_document(case['configuration'],case['times'],case['ne'],
                    'actual-basis','actual-gauge',basis)
                self.assertEqual(canonical_json(basis),case['basis_json'])
                self.assertEqual(canonical_json(values),case['eigenvalues_json'])
                self.assertEqual(wilson_reference._basis_spec(*args).definition_id,
                    DatasetSpec(json.loads(case['basis_json'])).definition_id)
                self.assertEqual(wilson_reference._values_spec(case['configuration'],case['times'],
                    case['ne'],'actual-basis','actual-gauge',DatasetSpec(basis)).definition_id,
                    DatasetSpec(json.loads(case['eigenvalues_json'])).definition_id)


if __name__=='__main__':
    unittest.main()
