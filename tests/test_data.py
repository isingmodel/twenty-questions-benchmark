from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twentyq.data import load_split, load_targets


class LoadTargetsTests(unittest.TestCase):
    def test_load_targets_normalizes_csv_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            targets_dir = Path(tmp)
            (targets_dir / "targets.csv").write_text(
                "id,name,domain,aliases,description\n"
                "object_widget,widget,objects,widget,A useful widget.\n",
                encoding="utf-8",
            )

            targets = load_targets(targets_dir)

        target = targets["object_widget"]
        self.assertEqual(target["domain"], "objects")
        self.assertEqual(target["aliases"], ["widget"])

    def test_load_targets_adds_canonical_name_to_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            targets_dir = Path(tmp)
            (targets_dir / "targets.csv").write_text(
                "id,name,domain,aliases,description\n"
                "object_widget,widget,objects,a widget,A useful widget.\n",
                encoding="utf-8",
            )

            targets = load_targets(targets_dir)

        target = targets["object_widget"]
        self.assertEqual(target["domain"], "objects")
        self.assertEqual(target["aliases"], ["widget", "a widget"])

    def test_load_targets_infers_domain_from_id_prefix_when_domain_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            targets_dir = Path(tmp)
            (targets_dir / "targets.csv").write_text(
                "id,name,domain,aliases,description\n"
                "object_widget,widget,,widget,A useful widget.\n",
                encoding="utf-8",
            )

            targets = load_targets(targets_dir)

        self.assertEqual(targets["object_widget"]["domain"], "objects")

    def test_load_targets_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            targets_dir = Path(tmp)
            (targets_dir / "targets.csv").write_text(
                "id,name,domain,aliases\n"
                "object_widget,widget,objects,widget\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Missing required CSV columns"):
                load_targets(targets_dir)

    def test_load_targets_rejects_empty_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            targets_dir = Path(tmp)
            (targets_dir / "targets.csv").write_text(
                "id,name,domain,aliases,description\n"
                "object_widget,widget,objects, ,A useful widget.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Expected at least one value in 'aliases'"):
                load_targets(targets_dir)

    def test_load_targets_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            targets_dir = Path(tmp)
            (targets_dir / "targets.csv").write_text(
                "id,name,domain,aliases,description\n"
                "object_widget,widget,objects,widget,A useful widget.\n"
                "object_widget,other widget,objects,other widget,Another widget.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate target id"):
                load_targets(targets_dir)


class LoadSplitTests(unittest.TestCase):
    def test_load_split_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp) / "test.txt"
            split_path.write_text("object_widget\nobject_widget\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate target id"):
                load_split(split_path, {"object_widget": {"id": "object_widget"}})
