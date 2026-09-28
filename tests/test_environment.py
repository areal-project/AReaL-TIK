"""Environment identity checks for resumed timing comparisons."""

import copy
import tempfile
import unittest
from pathlib import Path

from tools.environment import comparison_identity, cutlass_identity


class EnvironmentTests(unittest.TestCase):
    def test_header_hash_binds_content_and_relative_path(self):
        with tempfile.TemporaryDirectory() as temp:
            include = Path(temp)
            (include / 'cute').mkdir()
            header = include / 'cute/test.hpp'
            header.write_text('original')
            original = cutlass_identity(include)['header_tree_sha256']
            header.write_text('modified')
            self.assertNotEqual(original, cutlass_identity(include)['header_tree_sha256'])
            header.write_text('original')
            header.rename(include / 'cute/renamed.hpp')
            self.assertNotEqual(original, cutlass_identity(include)['header_tree_sha256'])

    def test_dependency_change_is_visible_but_installation_path_is_not(self):
        environment = {'torch': 'recorded', 'device_uuid': 'device-a', 'packages': {'FlashInfer': 'recorded'},
                       'baseline_modules': {'flashinfer': {'file': '/first/path', 'sha256': 'abc'}},
                       'cutlass': {'include_directory': '/first/include', 'header_tree_sha256': 'def'}}
        other = copy.deepcopy(environment)
        other['baseline_modules']['flashinfer']['file'] = '/second/path'
        other['cutlass']['include_directory'] = '/second/include'
        self.assertEqual(comparison_identity(environment), comparison_identity(other))
        other['packages']['FlashInfer'] = 'changed'
        self.assertNotEqual(comparison_identity(environment), comparison_identity(other))
