import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import sobolev_visualization as sv


class SobolevVisualizationTests(unittest.TestCase):
    def test_coordinate_frame_roundtrip_preserves_metadata_and_assets(self):
        coords = np.array(
            [
                [0.0, 1.0, 2.0],
                [3.5, 4.5, 5.5],
            ],
            dtype=np.float64,
        )
        payload = sv.encode_coordinate_frame(
            coords,
            frame_index=17,
            time_seconds=2.25,
            asset_ids=np.array([4, 9], dtype=np.int64),
        )

        frame = sv.decode_coordinate_frame(payload)

        self.assertEqual(frame.frame_index, 17)
        self.assertAlmostEqual(frame.time_seconds, 2.25)
        self.assertEqual(frame.n_nodes, 2)
        np.testing.assert_allclose(frame.coords, coords.astype(np.float32))
        np.testing.assert_array_equal(frame.asset_ids, np.array([4, 9], dtype=np.uint32))

    def test_coordinate_frame_rejects_bad_shapes_and_payloads(self):
        with self.assertRaises(ValueError):
            sv.encode_coordinate_frame(np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            sv.encode_coordinate_frame(np.zeros((2, 3)), asset_ids=[1])
        with self.assertRaises(ValueError):
            sv.decode_coordinate_frame(b"not-a-frame")

    def test_instance_matrices_translate_and_scale_nodes(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

        matrices = sv.instance_matrices(coords, scales=[2.0, 3.0])

        self.assertEqual(matrices.shape, (2, 4, 4))
        np.testing.assert_allclose(matrices[0, :3, 3], coords[0])
        np.testing.assert_allclose(matrices[1, :3, 3], coords[1])
        np.testing.assert_allclose(np.diag(matrices[0, :3, :3]), [2.0, 2.0, 2.0])
        np.testing.assert_allclose(np.diag(matrices[1, :3, :3]), [3.0, 3.0, 3.0])
        np.testing.assert_allclose(matrices[:, 3, :], [[0.0, 0.0, 0.0, 1.0]] * 2)

    def test_coarse_grain_far_field_preserves_active_nodes_and_groups_far_nodes(self):
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [22.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        result = sv.coarse_grain_far_field(
            coords,
            active_center=[0.0, 0.0, 0.0],
            active_radius=1.5,
            group_size=2,
        )

        self.assertEqual(result.source_indices, ((0,), (1,), (2, 3), (4, 5)))
        np.testing.assert_allclose(result.coords[0], coords[0])
        np.testing.assert_allclose(result.coords[1], coords[1])
        np.testing.assert_allclose(result.coords[2], [11.0, 0.0, 0.0])
        np.testing.assert_allclose(result.coords[3], [21.0, 0.0, 0.0])
        np.testing.assert_array_equal(result.is_coarse, [False, False, True, True])
        self.assertEqual(result.n_original, 6)

    def test_coarse_grain_far_field_rejects_invalid_controls(self):
        coords = np.zeros((2, 3), dtype=np.float64)

        with self.assertRaises(ValueError):
            sv.coarse_grain_far_field(coords, active_center=[0.0, 0.0], active_radius=1.0)
        with self.assertRaises(ValueError):
            sv.coarse_grain_far_field(coords, active_center=[0.0, 0.0, 0.0], active_radius=-1.0)
        with self.assertRaises(ValueError):
            sv.coarse_grain_far_field(coords, active_center=[0.0, 0.0, 0.0], active_radius=1.0, group_size=0)

    def test_asset_id_vector_from_counts_is_deterministic_for_mappings(self):
        ids = sv.asset_id_vector_from_counts({7: 2, 3: 1})

        np.testing.assert_array_equal(ids, np.array([3, 7, 7], dtype=np.uint32))

    def test_asset_id_vector_from_counts_rejects_negative_records(self):
        with self.assertRaises(ValueError):
            sv.asset_id_vector_from_counts([(1, -1)])
        with self.assertRaises(ValueError):
            sv.asset_id_vector_from_counts([(-1, 1)])

    def test_write_scene_manifest_records_multiscale_inputs(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scene.json"

            sv.write_scene_manifest(
                path,
                tomography_map="emdb/sample.mrc",
                segmentation_mesh="mesh/cell.usd",
                coordinate_frame="frames/frame_0001.bin",
                assets=[
                    sv.AssetPrototype(
                        asset_id=2,
                        name="ribosome",
                        path="assets/ribosome.usd",
                        source="PDB",
                        copy_number=3,
                    )
                ],
                sources={"macro": "CryoET Data Portal", "abundance": "PaxDB"},
            )

            text = path.read_text(encoding="utf-8")

        self.assertIn('"schema": "sobolev-whole-cell-scene-v1"', text)
        self.assertIn('"tomography_map": "emdb/sample.mrc"', text)
        self.assertIn('"copy_number": 3', text)
        self.assertIn('"CryoET Data Portal"', text)

    def test_write_usda_point_instancer_emits_prototypes_and_positions(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 2.5, 3.5]], dtype=np.float64)
        ids = np.array([4, 9], dtype=np.uint32)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scene.usda"
            sv.write_usda_point_instancer(
                path,
                coords,
                ids,
                [
                    sv.AssetPrototype(4, "80S ribosome", "assets/ribosome.usd"),
                    sv.AssetPrototype(9, "Actin monomer", "assets/actin.usd"),
                ],
                scene_name="Cell Section",
            )
            text = path.read_text(encoding="utf-8")

        self.assertIn('defaultPrim = "Cell_Section"', text)
        self.assertIn('def PointInstancer "Instances"', text)
        self.assertIn("</Cell_Section/Prototypes/A_80S_ribosome>", text)
        self.assertIn("int[] protoIndices = [0, 1]", text)
        self.assertIn("(1.5, 2.5, 3.5)", text)

    def test_write_usda_point_instancer_rejects_missing_prototype(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scene.usda"
            with self.assertRaises(ValueError):
                sv.write_usda_point_instancer(
                    path,
                    np.zeros((1, 3)),
                    [5],
                    [sv.AssetPrototype(4, "known")],
                )


if __name__ == "__main__":
    unittest.main()
