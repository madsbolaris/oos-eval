import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_kev_dataset


class BuildKevDatasetTests(unittest.TestCase):
    def test_build_preserves_native_splits_and_expected_counts(self):
        spec, partitions = build_kev_dataset.build()

        self.assertEqual(
            {name: len(rows) for name, rows in partitions.items()},
            {"train": 1600, "calibration": 160, "development": 160, "test": 480},
        )
        self.assertEqual(spec["domain"], "travel")
        self.assertTrue(all(row["_meta"]["original_split"] in {"test", "oos_test"} for row in partitions["test"]))

    def test_every_partition_contains_every_label(self):
        spec, partitions = build_kev_dataset.build()
        expected = set(json.loads((ROOT / "data" / "domains.json").read_text())["travel"])
        expected.add("oos")

        for rows in partitions.values():
            counts = {}
            for row in rows:
                label = row["questions"][spec["question"]["id"]]["label"]
                counts[label] = counts.get(label, 0) + 1
            actual = set(counts)
            self.assertEqual(actual, expected)
            self.assertEqual(len(set(counts.values())), 1)

    def test_rendered_files_are_reproducible(self):
        spec, partitions = build_kev_dataset.build()
        first = build_kev_dataset.render(spec, partitions)
        second = build_kev_dataset.render(spec, partitions)

        self.assertEqual(first, second)
        manifest = json.loads(first["manifest.json"])
        self.assertEqual(manifest["locked"], ["test"])
        self.assertEqual(manifest["source"]["revision"], "828f8093932c8fe6ca7936c3d2e52903b1c523de")

    def test_check_accepts_fresh_output(self):
        spec, partitions = build_kev_dataset.build()
        files = build_kev_dataset.render(spec, partitions)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            build_kev_dataset.write(output, files)
            build_kev_dataset.check(output, files)


if __name__ == "__main__":
    unittest.main()