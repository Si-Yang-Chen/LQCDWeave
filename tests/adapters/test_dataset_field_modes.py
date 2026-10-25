"""Module tests split from test_production_dataset.py; shared numerical fixtures stay unchanged."""

from unittest.mock import patch

from test_production_dataset import (
    DatasetReader,
    DatasetSpec,
    _ProductionDatasetFixture,
    axis,
    np,
)
from lqcdweave.adapters.production_dataset import (
    DatasetFieldHandle,
    DatasetPointHandle,
)


class ProductionDatasetTests(_ProductionDatasetFixture):
    def test_registered_mode_blocks_follow_physical_time_and_mode_order(self):
        with DatasetReader(self.root/"basis") as dataset:
            handle = DatasetFieldHandle(dataset)
            block = handle.read_modes("cfg", 2, 1, 3)
            np.testing.assert_array_equal(block,self.vectors[2,1:3])
            self.assertFalse(block.flags.writeable)
            with self.assertRaisesRegex(ValueError,"configuration key"):
                handle.read_modes("other",2,1,3)
            with self.assertRaisesRegex(ValueError,"invalid physical time or mode"):
                handle.read_modes("cfg",2,1,4)

    def test_field_and_point_loads_defer_values_and_push_selections_to_reader(self):
        with DatasetReader(self.root/"basis") as dataset:
            handle = DatasetFieldHandle(dataset)
            with patch.object(dataset, "read", wraps=dataset.read) as read:
                field = handle.load("cfg")
                self.assertEqual(read.call_count, 0)
                selected = field[2, [2, 0]]
                np.testing.assert_array_equal(selected, self.vectors[2, [2, 0]])
                self.assertEqual(read.call_count, 1)
                self.assertEqual(read.call_args.kwargs["time"], 2)
                self.assertEqual(read.call_args.kwargs["select"]["mode"], [9, 4])

        points = np.asarray([
            [[0, 0, 0], [2, 1, 1], [1, 0, 1]],
            [[1, 0, 0], [0, 1, 1], [2, 1, 0]],
            [[2, 0, 0], [1, 1, 0], [0, 0, 1]],
        ], dtype=np.int64)
        spec = DatasetSpec(dict(
            schema=1, kind="spatial_points", configuration=self.cfg,
            axes=[axis("time", [2, 0, 1]), axis("point", [4, 9, 1]),
                  axis("xyz", ["x", "y", "z"])],
            dtype="int64", bases={}, parents=[], definition={"selection": "fixture"},
            conventions={"representation": "coordinate_delta", "normalization": "unit_delta"},
        ))
        self.save("points", points[[2, 0, 1]], spec)
        with DatasetReader(self.root/"points") as dataset:
            handle = DatasetPointHandle(dataset)
            with patch.object(dataset, "read", wraps=dataset.read) as read:
                point_field = handle.load("cfg")
                self.assertEqual(read.call_count, 0)
                selected = point_field[0:2, :2, :]
                np.testing.assert_array_equal(selected, points[[0, 1], :2, :])
                self.assertEqual([call.kwargs["time"] for call in read.call_args_list], [0, 1])
                self.assertEqual(read.call_args_list[0].kwargs["select"]["point"], [4, 9])
                paired = point_field[[0, 2], [1, 1], :]
                np.testing.assert_array_equal(paired, points[[0, 2], [1, 1], :])
            dataset.close()
            with self.assertRaisesRegex(ValueError, "closed"):
                point_field[0, 0]
