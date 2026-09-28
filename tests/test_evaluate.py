"""CPU tests for independent audit reports; synthetic evidence is not GPU validation."""

import copy
import io
import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml
from tools import evaluate as evaluator

ROOT = Path(__file__).resolve().parents[1]


def config_for(gpu='h200'):
    return yaml.safe_load((ROOT / 'experimental' / gpu / 'config.yaml').read_text())


def measured_fixture(config, candidate=1.0, incumbent=1.0):
    """Construct a synthetic complete measurement record from the published grid."""
    spec = config['evaluation_spec']
    batches = sorted({cell['batch'] for cell in spec['workloads']})
    prompts = sorted({cell['length'] for cell in spec['workloads'] if cell['phase'] == 'prefill'})
    checks = []
    lengths = sorted(set(prompts + spec['raw_extra_lengths']))
    for (seed, scale), batch, length in itertools.product(spec['raw_seeds_scales'], batches, lengths):
        checks.append(dict(kind='raw', seed=seed, scale=scale, batch=batch, length=length,
                           passed=True, changed_input_graph_executed=length <= 8192))
    for batch, prompt, answer in itertools.product(batches, prompts, spec['partition_answer_lengths']):
        checks.append(dict(kind='partition', batch=batch, prompt=prompt, answer=answer, passed=True))
    if config['gpu']['name'] == 'A100':
        lengths = sorted({cell['length'] for cell in spec['workloads'] if cell['phase'] == 'decode'})
        for batch, length in itertools.product(batches, lengths):
            checks.append(dict(kind='batch', batch=batch, length=length, passed=True))
    rows = [dict(**cell, raw_equal=True, changed_input_graph_pass=True,
                 external_max_abs=0.001, cuda_dispatch=['synthetic_cuda_kernel'],
                 median_ms=dict(candidate=candidate, incumbent=incumbent, external=0.5, hopper_fa3=0.6))
            for cell in spec['workloads']]
    for row in rows:
        if row['phase'] == 'decode':
            row.update(external_name='FlashInfer[fa2,TC,test]',
                       hopper_fa3_name='FlashInfer[fa3,TC,test]', hopper_fa3_max_abs=0.001,
                       flashinfer_screen={'max_abs_less_than': spec['external_max_abs_less_than'],
                           'valid': [{'name': f'FlashInfer[{backend},TC,test]',
                                      'max_abs': 0.001, 'screen_ms': 0.5} for backend in ('fa2', 'fa3')],
                           'failures': []})
    return dict(verified=True, rows=rows, checks=checks)


class ReportTests(unittest.TestCase):
    def test_complete_reports_without_root_anchor_have_no_promotion_decision(self):
        for gpu in ('h20', 'h200', 'a100'):
            with self.subTest(gpu=gpu):
                config = config_for(gpu)
                report = measured_fixture(config, 0.9)
                result = evaluator.summarize_report(config, report)
                self.assertAlmostEqual(result['objective_ms'], 48 * 0.9)
                self.assertEqual(result['incumbent_objective_ms'], 48)
                self.assertIsNone(result['admissible'])
                self.assertIsNone(result['promotion_eligible'])

    def test_missing_or_invalid_evidence_cannot_qualify(self):
        for gpu in ('h200', 'a100'):
            config = config_for(gpu)
            cases = ('cell', 'partition', 'raw', 'duplicate', 'duplicate_check',
                     'nan', 'zero', 'bits', 'graph', 'dispatch', 'external', 'failed')
            cases += ('hopper',) if gpu == 'h200' else ('batch',)
            for case in cases:
                with self.subTest(gpu=gpu, case=case):
                    report = measured_fixture(config)
                    if case == 'cell':
                        report['rows'].pop()
                    elif case in ('raw', 'partition', 'batch'):
                        report['checks'] = [check for check in report['checks'] if check['kind'] != case]
                    elif case == 'duplicate':
                        report['rows'][-1] = copy.deepcopy(report['rows'][0])
                    elif case == 'duplicate_check':
                        report['checks'].append(copy.deepcopy(report['checks'][0]))
                    elif case in ('nan', 'zero'):
                        report['rows'][0]['median_ms']['candidate'] = float('nan') if case == 'nan' else 0
                    elif case == 'bits':
                        report['rows'][0]['raw_equal'] = False
                    elif case == 'graph':
                        report['rows'][0]['changed_input_graph_pass'] = False
                    elif case == 'dispatch':
                        report['rows'][0]['cuda_dispatch'] = []
                    elif case == 'external':
                        report['rows'][0]['external_max_abs'] = config['evaluation_spec']['external_max_abs_less_than']
                    elif case == 'hopper':
                        next(row for row in report['rows'] if row['phase'] == 'decode')['median_ms'].pop('hopper_fa3')
                    else:
                        report['verified'] = False
                    with self.assertRaises(ValueError):
                        evaluator.validate_report(config, report)

    def test_workload_regression_blocks_an_improved_aggregate(self):
        config = config_for()
        root = measured_fixture(config)
        candidate = measured_fixture(config, 0.9)
        candidate['rows'][0]['median_ms']['candidate'] = 1.06
        result = evaluator.summarize_report(config, candidate, root)
        self.assertLess(result['objective_ms'], result['incumbent_objective_ms'])
        self.assertFalse(result['admissible'])
        self.assertFalse(result['promotion_eligible'])
        self.assertEqual(result['slowdown_violations'], [candidate['rows'][0]['id']])
        self.assertTrue(candidate['verified'])

    def test_invalid_auxiliary_hopper_route_cannot_qualify(self):
        config = config_for()
        for failure in ('nan_error', 'missing_error', 'wrong_route', 'nan_screen', 'wrong_tolerance'):
            with self.subTest(failure=failure):
                report = measured_fixture(config)
                row = next(r for r in report['rows'] if r['phase'] == 'decode')
                if failure == 'nan_error':
                    row['hopper_fa3_max_abs'] = float('nan')
                elif failure == 'missing_error':
                    row.pop('hopper_fa3_max_abs')
                elif failure == 'wrong_route':
                    row['hopper_fa3_name'] = 'never-qualified'
                elif failure == 'nan_screen':
                    row['flashinfer_screen']['valid'][1]['max_abs'] = float('nan')
                else:
                    row['flashinfer_screen']['max_abs_less_than'] = 1.0
                with self.assertRaises(ValueError):
                    evaluator.validate_report(config, report)

    def test_weighted_latency_controls_promotion(self):
        config = config_for()
        config['evaluation_spec']['workloads'][0]['coefficient'] = 10000
        root = measured_fixture(config)
        candidate = measured_fixture(config, 0.5)
        candidate['rows'][0]['median_ms']['candidate'] = 1.04
        result = evaluator.summarize_report(config, candidate, root)
        self.assertTrue(result['admissible'])
        self.assertGreater(result['objective_ms'], result['incumbent_objective_ms'])
        self.assertFalse(result['promotion_eligible'])

    def test_equal_or_slower_verified_candidate_is_not_promoted(self):
        config = config_for()
        root = measured_fixture(config)
        for timing in (1.0, 1.01):
            with self.subTest(timing=timing):
                result = evaluator.summarize_report(config, measured_fixture(config, timing), root)
                self.assertTrue(result['admissible'])
                self.assertFalse(result['promotion_eligible'])

    def test_root_audit_requires_matching_gpu_config_sources_and_tools(self):
        config = config_for()
        report = dict(gpu='h200', root_sha256='root', config_sha256='config', tool_sha256={'tool': 'digest'})
        root = dict(measured_fixture(config), gpu='h200', source_sha256='root', root_sha256='root',
                    incumbent_sha256='root', config_sha256='config', tool_sha256={'tool': 'digest'})
        evaluator.validate_root_report(config, report, root)
        for field in ('gpu', 'config_sha256', 'source_sha256', 'root_sha256', 'incumbent_sha256', 'tool_sha256'):
            with self.subTest(field=field):
                wrong = copy.deepcopy(root)
                wrong[field] = 'wrong'
                with self.assertRaises(ValueError):
                    evaluator.validate_root_report(config, report, wrong)


class EvaluatorCliTests(unittest.TestCase):
    def test_gpu_defaults_save_complete_source_bound_reports(self):
        for gpu in ('h20', 'h200', 'a100'):
            with self.subTest(gpu=gpu), tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
                output = Path(temp) / 'reports/evaluation.json'

                def fake_execute(args, report):
                    self.assertEqual(args.config, ROOT / 'experimental' / gpu / 'config.yaml')
                    self.assertEqual(args.source, ROOT / 'experimental' / gpu / 'kernels/unified_attn_ext.cu')
                    self.assertEqual(args.root, args.source)
                    self.assertEqual(args.incumbent, args.source)
                    report.update(measured_fixture(config_for(gpu)))

                with patch.object(evaluator, 'execute', side_effect=fake_execute):
                    self.assertEqual(evaluator.main(['--gpu', gpu, '--output', str(output)]), 0)
                report = json.loads(output.read_text())
                self.assertTrue(report['verified'])
                self.assertEqual(report['gpu'], gpu)
                self.assertEqual(report['source_sha256'], report['root_sha256'])
                self.assertEqual(set(report['tool_sha256']), set(evaluator.TOOL_FILES))
                self.assertGreaterEqual(report['elapsed_s'], 0)

    def test_frozen_run_root_then_candidate_with_paired_incumbent(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            run = Path(temp)
            config_path = run / 'config.yaml'
            config_path.write_bytes((ROOT / 'experimental/h200/config.yaml').read_bytes())
            config = yaml.safe_load(config_path.read_text())
            root_source, candidate_source = run / 'root.cu', run / 'candidate.cu'
            root_source.write_bytes((ROOT / 'experimental/h200/kernels/unified_attn_ext.cu').read_bytes())
            candidate_source.write_text(root_source.read_text() + '\n// CPU test fixture\n')
            root_output, candidate_output = run / 'root.json', run / 'candidate.json'
            common = ['--config', str(config_path), '--root', str(root_source), '--incumbent', str(root_source)]
            with patch.object(evaluator, 'execute', side_effect=lambda args, report: report.update(measured_fixture(config))):
                self.assertEqual(evaluator.main(common + ['--source', str(root_source), '--output', str(root_output)]), 0)
            root_bytes = root_output.read_bytes()
            with patch.object(evaluator, 'execute', side_effect=lambda args, report: report.update(measured_fixture(config, 0.9))):
                self.assertEqual(evaluator.main(common + ['--source', str(candidate_source), '--root-report',
                                                        str(root_output), '--output', str(candidate_output)]), 0)
            report = json.loads(candidate_output.read_text())
            self.assertTrue(report['verified'])
            self.assertTrue(report['admissible'])
            self.assertTrue(report['promotion_eligible'])
            self.assertEqual(report['root_report_sha256'], evaluator.sha(root_output))
            self.assertEqual(report['source_sha256'], evaluator.sha(candidate_source))
            self.assertEqual(root_output.read_bytes(), root_bytes)
            self.assertFalse((run / 'best.cu').exists())  # Evaluation never edits search state.
            self.assertFalse((run / 'state.json').exists())

    def test_unrelated_root_report_is_rejected_before_gpu_execution(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root_path, output = Path(temp) / 'wrong-root.json', Path(temp) / 'failure.json'
            root_path.write_text(json.dumps(dict(measured_fixture(config_for()), gpu='a100')))
            with patch.object(evaluator, 'execute') as execute:
                self.assertEqual(evaluator.main(['--gpu', 'h200', '--root-report', str(root_path),
                                                 '--output', str(output)]), 1)
                execute.assert_not_called()
            report = json.loads(output.read_text())
            self.assertFalse(report['verified'])
            self.assertFalse(report['promotion_eligible'])

    def test_a_success_flag_without_measurements_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp) / 'empty.json'
            with patch.object(evaluator, 'execute', side_effect=lambda args, report: report.update(verified=True)):
                self.assertEqual(evaluator.main(['--gpu', 'h200', '--output', str(output)]), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report['verified'])
            self.assertIn('workload grid', report['error'])

    def test_changed_config_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            config_path, output = Path(temp) / 'config.yaml', Path(temp) / 'changed.json'
            config_path.write_bytes((ROOT / 'experimental/h200/config.yaml').read_bytes())

            def fake_execute(args, report):
                report.update(measured_fixture(config_for()))
                config_path.write_text(config_path.read_text() + '\n# changed during evaluation\n')

            with patch.object(evaluator, 'execute', side_effect=fake_execute):
                self.assertEqual(evaluator.main(['--config', str(config_path), '--output', str(output)]), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report['verified'])
            self.assertIn('config changed', report['error'])

    def test_failed_execution_saves_failure_and_nonzero_status(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp) / 'failure.json'
            with patch.object(evaluator, 'execute', side_effect=RuntimeError('CUDA fixture failed')):
                self.assertEqual(evaluator.main(['--gpu', 'h200', '--output', str(output)]), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report['verified'])
            self.assertFalse(report['promotion_eligible'])
            self.assertIn('CUDA fixture failed', report['error'])
            self.assertIn('traceback', report)

    def test_existing_results_and_mismatched_gpu_are_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stderr(io.StringIO()):
            output = Path(temp) / 'existing.json'
            output.write_text('preserved result')
            with patch.object(evaluator, 'execute') as execute:
                with self.assertRaises(SystemExit):
                    evaluator.main(['--gpu', 'h200', '--output', str(output)])
                with self.assertRaises(SystemExit):
                    evaluator.main(['--gpu', 'a100', '--config', str(ROOT / 'experimental/h200/config.yaml')])
                execute.assert_not_called()
            self.assertEqual(output.read_text(), 'preserved result')


if __name__ == '__main__':
    unittest.main()
