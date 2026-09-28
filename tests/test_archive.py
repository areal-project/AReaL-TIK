"""Reject altered evidence and figures without needing GPU execution."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import verify_archive as archive


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = archive.load_evidence(archive.ROOT / 'experimental/evidence')

    def test_all_six_figure_values_follow_original_measurements(self):
        archive.verify_figure(archive.ROOT / 'experimental/figures', self.records)

    def test_changed_figure_latency_is_rejected(self):
        original = archive.ROOT / 'experimental/figures/attention_performance_cells.csv'
        with original.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['candidate_ms'] = str(float(rows[0]['candidate_ms']) * 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cells.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            resolve = archive.internal_path
            def replacement(root, name):
                return path if name.endswith('attention_performance_cells.csv') else resolve(root, name)
            with patch.object(archive, 'internal_path', side_effect=replacement), self.assertRaises(ValueError):
                archive.verify_figure(archive.ROOT / 'experimental/figures', self.records)

    def test_unknown_hash_and_wrong_origin_are_rejected(self):
        with self.assertRaises(ValueError):
            archive.verify_reference({'record_sha256': 'missing', 'record': 'missing'}, self.records)
        sha = next(iter(self.records))
        with self.assertRaises(ValueError):
            archive.verify_reference({'record_sha256': sha, 'record': 'wrong-origin'}, self.records)
