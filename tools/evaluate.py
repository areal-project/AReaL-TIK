"""Check attention bitwise alignment and benchmark against FlashAttention/FlashInfer.

Run: python tools/evaluate.py --gpu h200 --output runs/h200-evaluation.json
The config defines the correctness suites and matched CUDA-graph timing grid.
Explicit source/root/incumbent arguments also support the agent's independent audit.
"""

import argparse
import gc
import hashlib
import json
import math
import re
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOL_FILES = ('tools/evaluate.py', 'tools/baselines.py', 'tools/load_kernel.py', 'tools/environment.py')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_checks(config):
    spec = config['evaluation_spec']
    batches = sorted({c['batch'] for c in spec['workloads']})
    prompts = sorted({c['length'] for c in spec['workloads'] if c['phase'] == 'prefill'})
    lengths = sorted(set(prompts + spec['raw_extra_lengths']))
    expected = {('raw', seed, scale, b, n) for seed, scale in spec['raw_seeds_scales']
                for b in batches for n in lengths}
    expected |= {('partition', b, n, a) for b in batches for n in prompts for a in spec['partition_answer_lengths']}
    if config['gpu']['name'] == 'A100':
        expected |= {('batch', b, n) for b in batches for n in
                     {c['length'] for c in spec['workloads'] if c['phase'] == 'decode'}}
    return expected


def validate_report(config, report):
    require(report.get('verified') is True, report.get('error', 'Evaluator did not verify this source'))
    spec = config['evaluation_spec']
    cells = {c['id']: c for c in spec['workloads']}
    require(spec['objective'] == 'weighted_sum_ms', 'Unsupported objective')
    require(bool(cells) and len(cells) == len(spec['workloads']), 'Invalid configured workload grid')
    for cell in cells.values():
        require(math.isfinite(cell['coefficient']) and cell['coefficient'] > 0, 'Invalid workload weight')
        require(math.isfinite(cell['max_slowdown_ratio']) and cell['max_slowdown_ratio'] >= 1,
                'Invalid workload slowdown cap')
    rows = report.get('rows', [])
    require(len(rows) == len(cells) and {r['id'] for r in rows} == set(cells), 'Incomplete or duplicated workload grid')
    for row in rows:
        cell = cells[row['id']]
        require(all(row[k] == v for k, v in cell.items()), 'Workload specification changed')
        require(row.get('raw_equal') is True and row.get('changed_input_graph_pass') is True, 'Raw-bit gate failed')
        require(bool(row.get('cuda_dispatch')), 'Missing CUDA dispatch evidence')
        err = row['external_max_abs']
        require(math.isfinite(err) and 0 <= err < spec['external_max_abs_less_than'], 'External numerical gate failed')
        for key in ('candidate', 'incumbent', 'external'):
            value = row['median_ms'].get(key)
            require(isinstance(value, (float, int)) and math.isfinite(value) and value > 0, 'Invalid/missing timing')
        if row['phase'] == 'decode' and spec['require_flashinfer_hopper_fa3']:
            value = row['median_ms'].get('hopper_fa3')
            require(isinstance(value, (float, int)) and math.isfinite(value) and value > 0,
                    'Missing/invalid Hopper FA3 timing')
            error = row.get('hopper_fa3_max_abs')
            require(isinstance(error, (float, int)) and math.isfinite(error)
                    and 0 <= error < spec['external_max_abs_less_than'],
                    'Missing/invalid Hopper FA3 numerical qualification')
        if row['phase'] == 'decode':
            screen = row.get('flashinfer_screen') or {}
            require(screen.get('max_abs_less_than') == spec['external_max_abs_less_than'],
                    'FlashInfer screening tolerance mismatch')
            qualified = screen.get('valid', [])
            require(bool(qualified), 'Missing FlashInfer screening evidence')
            for route in qualified:
                error, latency = route.get('max_abs'), route.get('screen_ms')
                require(isinstance(error, (float, int)) and math.isfinite(error)
                        and 0 <= error < spec['external_max_abs_less_than'], 'Invalid FlashInfer route error')
                require(isinstance(latency, (float, int)) and math.isfinite(latency) and latency > 0,
                        'Invalid FlashInfer screen latency')
            names = {route['name'] for route in qualified}
            require(row.get('external_name') in names, 'Selected FlashInfer route was not qualified')
            if spec['require_flashinfer_hopper_fa3']:
                require(row.get('hopper_fa3_name') in names and '[fa3,TC,' in row['hopper_fa3_name'],
                        'Hopper comparison route was not qualified')
    observed = set()
    for check in report.get('checks', []):
        require(check.get('passed') is True, 'Expanded correctness failure')
        if check['kind'] == 'raw':
            require(check['changed_input_graph_executed'] == (check['length'] <= 8192), 'Graph-check coverage mismatch')
            key = ('raw', check['seed'], check['scale'], check['batch'], check['length'])
        elif check['kind'] == 'partition':
            key = ('partition', check['batch'], check['prompt'], check['answer'])
        elif check['kind'] == 'batch':
            key = ('batch', check['batch'], check['length'])
        else:
            raise ValueError('Unexpected check type')
        require(key not in observed, 'Duplicate expanded check')
        observed.add(key)
    require(observed == expected_checks(config), 'Missing/extra expanded correctness cases')
    objective = sum(r['coefficient'] * r['median_ms']['candidate'] for r in rows)
    incumbent = sum(r['coefficient'] * r['median_ms']['incumbent'] for r in rows)
    require(math.isfinite(objective) and math.isfinite(incumbent), 'Nonfinite objective')
    return objective, incumbent


def summarize_report(config, report, root_report=None):
    """Validate measured evidence and compare it without modifying search records."""
    objective, incumbent = validate_report(config, report)
    summary = {'objective_ms': objective, 'incumbent_objective_ms': incumbent,
               'admissible': None, 'promotion_eligible': None}
    if root_report is not None:
        validate_report(config, root_report)
        root_rows = {row['id']: row['median_ms']['candidate'] for row in root_report['rows']}
        regressions = [row['id'] for row in report['rows']
                       if row['median_ms']['candidate'] >
                       root_rows[row['id']] * row['max_slowdown_ratio']]
        summary.update(admissible=not regressions, slowdown_violations=regressions,
                       promotion_eligible=not regressions and objective < incumbent)
    return summary


def validate_root_report(config, report, root_report):
    """Require a complete qualification audit from this frozen run and evaluator."""
    validate_report(config, root_report)
    require(root_report.get('gpu') == report['gpu'], 'Root report GPU mismatch')
    require(root_report.get('config_sha256') == report['config_sha256'], 'Root report config hash mismatch')
    require(root_report.get('tool_sha256') == report['tool_sha256'], 'Root report tool hashes mismatch')
    for key in ('source_sha256', 'root_sha256', 'incumbent_sha256'):
        require(root_report.get(key) == report['root_sha256'], 'Root report source hash mismatch')


def execute(args, report):
    import torch
    from torch.profiler import ProfilerActivity, profile
    from tools.load_kernel import load_source
    from tools import baselines
    from tools.environment import collect_environment, comparison_identity

    config = yaml.safe_load(args.config.read_text())
    spec = config['evaluation_spec']
    gpu = config['gpu']['name'].lower()
    report['environment'] = collect_environment(config)
    if args.root_report is not None:
        anchor = json.loads(args.root_report.read_text())
        require(comparison_identity(anchor.get('environment')) == comparison_identity(report['environment']),
                'Root audit environment differs; qualify a new root in a separate run')
    candidate = load_source(gpu, args.source, config)
    root = load_source(gpu, args.root, config)
    incumbent = load_source(gpu, args.incumbent, config)
    global_names = re.findall(r'__global__[\s\S]{0,240}?\bvoid\s+(\w+)\s*\(', args.source.read_text())
    if not global_names:
        raise RuntimeError('No candidate CUDA kernel definitions found')
    hq, hkv, dim = spec['query_heads'], spec['kv_heads'], spec['head_dim']

    def tensors(batch, length, decode=False, scale=1.0):
        q = torch.randn(batch, hq, 1 if decode else length, dim, device='cuda', dtype=torch.bfloat16) * scale
        k = torch.randn(batch, hkv, length, dim, device='cuda', dtype=torch.bfloat16) * scale
        return q, k, torch.randn_like(k)

    def equal(a, b, label):
        if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
            raise RuntimeError(f'Nonfinite output: {label}')
        if not torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)):
            raise RuntimeError(f'Raw BF16 mismatch: {label}')

    def call(module, phase, q, k, v):
        return getattr(module, 'unified_decode' if phase == 'decode' else 'unified_prefill')(q, k, v, hq, hkv)

    def flush():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    # Source-bound paired performance grid, including policy-update forward.
    torch.manual_seed(20260905)
    for cell in spec['workloads']:
        phase = cell['phase']
        length = cell['length'] + cell['answer']
        q, k, v = tensors(cell['batch'], length, phase == 'decode')
        expected = call(root, phase, q, k, v)
        run = lambda: call(candidate, phase, q, k, v)
        equal(run(), expected, cell['id'])
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run()
            torch.cuda.synchronize()
        dispatch = sorted({event.name for event in prof.events()
                           if event.device_type == torch.autograd.DeviceType.CUDA})
        if not any(any(name in event for name in global_names) for event in dispatch):
            raise RuntimeError(f'No native candidate CUDA dispatch recorded: {cell["id"]}')
        candidate_graph = baselines.capture_cuda_graph(run)
        q.neg_()
        k.mul_(0.5)
        v.add_(0.125)
        expected = call(root, phase, q, k, v)
        equal(candidate_graph(), expected, cell['id'] + ' changed-input graph')
        equal(call(incumbent, phase, q, k, v), expected, cell['id'] + ' incumbent')
        runners = {'candidate': candidate_graph,
                   'incumbent': baselines.capture_cuda_graph(lambda: call(incumbent, phase, q, k, v))}
        screen = None
        hopper_error = hopper_name = None
        if phase == 'decode':
            external, external_name = baselines.prepare_flashinfer_decode(
                q, k, v, True, 'graph', ['fa2', 'fa3'] if gpu != 'a100' else ['fa2'], expected,
                spec['external_max_abs_less_than'])
            screen = external.screen
            if spec['require_flashinfer_hopper_fa3'] and external.hopper is None:
                raise RuntimeError('No qualified FlashInfer Hopper FA3 Tensor Core baseline')
            if external.hopper is not None:
                runners['hopper_fa3'] = external.hopper[0]
                hopper_name = external.hopper[1]
                hopper_error = baselines.numerical_error(external.hopper[0](), expected,
                                                        spec['external_max_abs_less_than'])
        else:
            external_name = 'FA2' if gpu == 'a100' else 'FA3'
            external = baselines.capture_cuda_graph(
                lambda: baselines.fa_prefill(q, k, v, True, 2 if gpu == 'a100' else 3))
        oracle = external()
        error = baselines.numerical_error(oracle, expected, spec['external_max_abs_less_than'])
        runners['external'] = external
        samples = {key: [] for key in runners}
        repeats = 50 if phase == 'decode' else (15 if cell['batch'] * length * length < 2**29 else 5)
        for turn in range(spec['rounds']):
            order = list(runners)
            if turn % 2:
                order.reverse()
            for name in order:
                samples[name].append(baselines.cuda_timer(runners[name], 5, repeats)['median_ms'])
        row = {**cell, 'raw_equal': True, 'changed_input_graph_pass': True,
               'external_name': external_name, 'external_max_abs': error,
               'cuda_dispatch': dispatch, 'rounds_ms': samples,
               'median_ms': {name: statistics.median(values) for name, values in samples.items()},
               'flashinfer_screen': screen, 'hopper_fa3_name': hopper_name,
               'hopper_fa3_max_abs': hopper_error}
        if not all(math.isfinite(value) and value > 0 for value in row['median_ms'].values()):
            raise RuntimeError(f'Invalid latency: {cell["id"]}')
        report['rows'].append(row)
        flush()
        del runners, candidate_graph, external, expected, oracle, q, k, v, prof, run
        gc.collect()
        torch.cuda.empty_cache()

    # Retain the original multi-seed root-bit, exact-context and graph tests.
    batches = sorted({c['batch'] for c in spec['workloads']})
    prompts = sorted({c['length'] for c in spec['workloads'] if c['phase'] == 'prefill'})
    raw_lengths = sorted(set(prompts + spec['raw_extra_lengths']))
    for seed, scale in spec['raw_seeds_scales']:
        torch.manual_seed(seed)
        for batch in batches:
            for length in raw_lengths:
                q, k, v = tensors(batch, length, scale=scale)
                out = candidate.unified_prefill(q, k, v, hq, hkv)
                equal(out, root.unified_prefill(q, k, v, hq, hkv), f'raw {seed}/{batch}/{length}')
                last = q[:, :, -1:].contiguous()
                equal(out[:, :, -1:], candidate.unified_decode(last, k, v, hq, hkv), 'exact-context last row')
                if length <= 8192:
                    graph = baselines.capture_cuda_graph(lambda: candidate.unified_decode(last, k, v, hq, hkv))
                    last.neg_(); k.mul_(0.5); v.add_(0.125)
                    equal(graph(), root.unified_decode(last, k, v, hq, hkv), 'raw changed-input graph')
                    del graph
                report['checks'].append({'kind': 'raw', 'seed': seed, 'scale': scale,
                                         'batch': batch, 'length': length, 'passed': True,
                                         'changed_input_graph_executed': length <= 8192})
                flush()
                del out, q, k, v, last

    # Prompt-prefix equality and ALL answer tokens, including A100 partial tails.
    torch.manual_seed(1234)
    for batch in batches:
        for prompt in prompts:
            for answer in spec['partition_answer_lengths']:
                q, k, v = tensors(batch, prompt + answer)
                full = candidate.unified_prefill(q, k, v, hq, hkv)
                prefix = candidate.unified_prefill(q[:, :, :prompt].contiguous(),
                    k[:, :, :prompt].contiguous(), v[:, :, :prompt].contiguous(), hq, hkv)
                equal(full[:, :, :prompt], prefix, 'prompt prefix')
                for pos in range(prompt, prompt + answer):
                    step = candidate.unified_decode(q[:, :, pos:pos+1].contiguous(),
                        k[:, :, :pos+1].contiguous(), v[:, :, :pos+1].contiguous(), hq, hkv)
                    equal(full[:, :, pos:pos+1], step, 'answer partition')
                report['checks'].append({'kind': 'partition', 'batch': batch, 'prompt': prompt,
                                         'answer': answer, 'passed': True})
                flush()
                del q, k, v, full, prefix, step
    if gpu == 'a100':
        for batch in batches:
            for length in sorted({c['length'] for c in spec['workloads'] if c['phase'] == 'decode'}):
                q, k, v = tensors(batch, length, True)
                out = candidate.unified_decode(q, k, v, hq, hkv)
                singles = torch.cat([candidate.unified_decode(q[b:b+1], k[b:b+1], v[b:b+1], hq, hkv)
                                     for b in range(batch)], dim=0)
                equal(out, singles, 'decode batch consistency')
                report['checks'].append({'kind': 'batch', 'batch': batch, 'length': length, 'passed': True})
                flush()
    torch.cuda.synchronize()
    report['verified'] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=('h20', 'h200', 'a100'),
                        help='Evaluate the selected kernel for this GPU')
    parser.add_argument('--config', type=Path, help='Override the per-GPU evaluation config')
    parser.add_argument('--source', type=Path, help='Candidate CUDA source; defaults to the selected kernel')
    parser.add_argument('--root', type=Path, help='Bitwise reference source; defaults to the selected kernel')
    parser.add_argument('--incumbent', type=Path, help='Paired timing reference; defaults to --root')
    parser.add_argument('--root-report', type=Path,
                        help='Frozen root qualification JSON; enables slowdown-cap and promotion eligibility checks')
    parser.add_argument('--output', type=Path, help='New JSON result path; defaults to a timestamped file in runs/')
    args = parser.parse_args(argv)
    if args.config is None:
        if args.gpu is None:
            parser.error('Provide --gpu or --config')
        args.config = ROOT / 'experimental' / args.gpu / 'config.yaml'
    args.config = args.config.expanduser().resolve()
    if not args.config.is_file():
        parser.error(f'Config does not exist: {args.config}')
    config = yaml.safe_load(args.config.read_text())
    gpu = config['gpu']['name'].lower()
    if gpu not in ('h20', 'h200', 'a100') or (args.gpu is not None and args.gpu != gpu):
        parser.error('--gpu and config GPU must match a supported device')
    selected = ROOT / 'experimental' / gpu / config['kernel']['source']
    if args.source is None or args.root is None:
        if not selected.is_file() or sha(selected) != config['kernel']['sha256']:
            parser.error('Selected kernel is missing or its source hash has changed')
    args.source = args.source or selected
    args.root = args.root or selected
    args.incumbent = args.incumbent or args.root
    for name in ('source', 'root', 'incumbent'):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f'{name} source does not exist: {path}')
        setattr(args, name, path)
    if args.root_report is not None:
        args.root_report = args.root_report.expanduser().resolve()
        if not args.root_report.is_file():
            parser.error(f'Root report does not exist: {args.root_report}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    args.output = (args.output or ROOT / 'runs' / f'{gpu}-evaluation-{stamp}.json').expanduser().resolve()
    if args.output.exists():
        parser.error(f'Result already exists; choose a new output path: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'areal-tik-audit-v3', 'started_at': now(), 'verified': False,
              'gpu': gpu,
              'source_sha256': sha(args.source), 'config_sha256': sha(args.config),
              'root_sha256': sha(args.root), 'incumbent_sha256': sha(args.incumbent),
              'tool_sha256': {name: sha(ROOT / name) for name in TOOL_FILES},
              'root_report_sha256': sha(args.root_report) if args.root_report else None,
              'command': [sys.executable, str(Path(__file__).resolve()),
                          *(argv if argv is not None else sys.argv[1:])],
              'admissible': None, 'promotion_eligible': None,
              'rows': [], 'checks': [], 'token_usage': None}
    start = time.monotonic()
    try:
        root_report = None
        if args.root_report is not None:
            root_report = json.loads(args.root_report.read_text())
            validate_root_report(config, report, root_report)
        execute(args, report)
        for name in ('source', 'config', 'root', 'incumbent'):
            if sha(getattr(args, name)) != report[name + '_sha256']:
                raise RuntimeError(f'{name} changed during evaluation')
        if args.root_report is not None and sha(args.root_report) != report['root_report_sha256']:
            raise RuntimeError('Root report changed during evaluation')
        if report['tool_sha256'] != {name: sha(ROOT / name) for name in TOOL_FILES}:
            raise RuntimeError('Evaluator tools changed during evaluation')
        report.update(summarize_report(config, report, root_report))
    except Exception as error:
        report['verified'] = False
        report['promotion_eligible'] = False
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
    finally:
        report.update(finished_at=now(), elapsed_s=time.monotonic()-start)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'{"PASS" if report["verified"] else "FAIL"}: {len(report["rows"])} timed workloads; '
          f'{len(report["checks"])} additional correctness cases')
    if report['verified']:
        for phase in ('prefill', 'training_forward', 'decode'):
            rows = [row for row in report['rows'] if row['phase'] == phase]
            if rows:
                speedup = math.exp(statistics.mean(math.log(row['median_ms']['external'] /
                    row['median_ms']['candidate']) for row in rows))
                print(f'{phase}: {speedup:.3f}x averaged baseline/candidate ({len(rows)} workloads)')
        if args.root_report is not None:
            print(f'Admissible: {report["admissible"]}; promotion eligible: {report["promotion_eligible"]}')
    else:
        print(report.get('error', 'Correctness or benchmark evaluation failed'))
    print(f'Report: {args.output}')
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
