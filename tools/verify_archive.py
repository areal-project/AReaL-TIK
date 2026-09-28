"""Check source integrity, classification, config, and saved validation without a GPU."""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PASS_QUALIFICATIONS = {"recorded_correctness_pass", "selected_full_grid_pass"}
DEMO_QUALIFICATIONS = {"known_failure_or_limitation", "pending_or_insufficient_evidence"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def internal_path(directory, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, f"Invalid path: {name}")
    path = directory / relative
    require(not path.is_symlink(), f"Unexpected symlink: {path}")
    require(path.resolve().is_relative_to(directory.resolve()), f"Path escapes archive: {path}")
    require(path.is_file(), f"Missing file: {path}")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence(directory):
    index = json.loads((directory / 'index.json').read_text())
    archive = internal_path(directory, index['archive'])
    require(digest(archive) == index['archive_sha256'], 'Evidence archive hash mismatch')
    records = {}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        expected = {item['member'] for item in index['records'].values()}
        require(len(members) == len(expected) and {m.name for m in members} == expected,
                'Missing, duplicate or extra evidence members')
        for sha, item in index['records'].items():
            require(item['format'] in ('json', 'text'), 'Unknown evidence format')
            require(item['member'] == sha + ('.json' if item['format'] == 'json' else '.txt'),
                    'Invalid evidence member name')
            member = tar.getmember(item['member'])
            require(member.isfile() and member.size == item['bytes'], 'Invalid evidence member')
            data = tar.extractfile(member).read()
            require(hashlib.sha256(data).hexdigest() == sha, f'Evidence hash mismatch: {sha}')
            records[sha] = {'data': data, 'json': json.loads(data) if item['format'] == 'json' else None,
                            'original_records': item['original_records']}
    return records


def verify_reference(reference, records):
    sha = reference['record_sha256']
    require(sha in records, f'Missing packaged evidence: {sha}')
    require(reference['record'] in records[sha]['original_records'], 'Evidence origin mismatch')
    return records[sha]['json']


def verify_figure(directory, records):
    provenance = json.loads((directory / 'provenance.json').read_text())['performance']
    with internal_path(ROOT, provenance['cells_file']).open() as stream:
        cells = list(csv.DictReader(stream))
    with internal_path(ROOT, provenance['data_file']).open() as stream:
        bars = list(csv.DictReader(stream))
    expected = {(gpu, phase, b, n) for gpu in ('H20', 'H200', 'A100')
                for phase, batches, lengths in [('prefill', [1], [1024, 4096, 8192, 16384]),
                                                 ('decode', [1, 4, 8, 16], [128, 1024, 4096, 8192])]
                for b in batches for n in lengths}
    observed = {(r['gpu'], r['phase'], int(r['batch']), int(r['length'])) for r in cells}
    require(len(cells) == len(expected) and observed == expected, 'Incomplete/duplicate figure cells')
    for row in cells:
        source = provenance['source_records'][row['gpu']]
        require(row['record_sha256'] == source['record_sha256']
                and row['source_sha256'] == source['source_sha256'], 'Figure provenance mismatch')
        raw = records[row['record_sha256']]['json']
        require(raw['sources'][source['source_key']] == row['source_sha256'], 'Figure source mismatch')
        matches = [r for r in raw['rows'] if (r['phase'], r['batch'], r['length']) ==
                   (row['phase'], int(row['batch']), int(row['length']))]
        require(len(matches) == 1, 'Figure cell absent or ambiguous in original evidence')
        original = matches[0]
        candidate, external = float(row['candidate_ms']), float(row['external_ms'])
        require(all(math.isfinite(v) and v > 0 for v in (candidate, external)), 'Invalid figure latency')
        require(candidate == original['median_ms'][source['source_key']]
                and external == original['median_ms'][original['external_name']]
                and row['external'] == re.sub(r'^FlashInfer[^\[]*', 'FlashInfer', original['external_name']),
                'Figure timing differs from evidence')
        require(math.isclose(float(row['speedup']), external / candidate, rel_tol=1e-12),
                'Incorrect per-cell figure ratio')
    require(len(bars) == 6 and {(r['gpu'], r['phase']) for r in bars} ==
            {(gpu, phase) for gpu in ('H20', 'H200', 'A100') for phase in ('prefill', 'decode')},
            'Invalid figure summary grid')
    for bar in bars:
        group = [r for r in cells if (r['gpu'], r['phase']) == (bar['gpu'], bar['phase'])]
        result = math.exp(sum(math.log(float(r['external_ms']) / float(r['candidate_ms']))
                             for r in group) / len(group))
        require(len(group) == int(bar['cells']) and math.isclose(result, float(bar['speedup']), rel_tol=1e-12),
                'Figure aggregate differs from source measurements')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', metavar='SHA256', help='Print one original evidence record, after verifying its hash')
    args = parser.parse_args()
    directory_root = ROOT / "experimental"
    records = load_evidence(directory_root / 'evidence')
    if args.evidence:
        require(args.evidence in records, 'Unknown evidence hash')
        sys.stdout.buffer.write(records[args.evidence]['data'])
        return
    archive = json.loads((directory_root / "archive.json").read_text())
    total = 0
    for gpu, summary in archive["gpus"].items():
        directory = directory_root / gpu
        config = yaml.safe_load((directory / "config.yaml").read_text())
        manifest = json.loads(internal_path(directory, config["kernel"]["manifest"]).read_text())
        validation = json.loads(internal_path(directory, config["kernel"]["validation"]).read_text())
        require(config["gpu"]["name"].lower() == gpu == manifest["gpu"], f"GPU mismatch: {gpu}")
        hashes, paths, counts = set(), set(), Counter()
        for row in manifest["kernels"]:
            path = internal_path(directory, row["path"])
            require(digest(path) == row["sha256"], f"Hash mismatch: {path}")
            require(path.stat().st_size == row["bytes"], f"Size mismatch: {path}")
            require(row["sha256"] not in hashes, f"Duplicate source bytes: {path}")
            require(row["path"] not in paths, f"Duplicate manifest path: {path}")
            hashes.add(row["sha256"])
            paths.add(row["path"])
            category = Path(row["path"]).parent.name
            if category == "snapshots":
                require(row["qualification"] in PASS_QUALIFICATIONS, f"Unqualified snapshot: {path}")
                require(bool(row["evidence"]), f"Missing snapshot evidence: {path}")
                require(not row["limitations"], f"Known limitation in snapshots: {path}")
            else:
                require(category == "snapshots-demo", f"Unknown source category: {path}")
                require(row["qualification"] in DEMO_QUALIFICATIONS, f"Invalid demo status: {path}")
            counts[category] += 1
            for evidence in row['evidence']:
                verify_reference(evidence, records)

        selected = config["kernel"]
        require(selected["snapshot"] == manifest["selected"] == summary["selected_snapshot"]
                == validation["immutable_snapshot"], f"Selected path mismatch: {gpu}")
        require(selected["source"] == validation["selected_source"], f"Active source mismatch: {gpu}")
        source = internal_path(directory, selected["source"])
        snapshot = internal_path(directory, selected["snapshot"])
        require(digest(source) == digest(snapshot) == selected["sha256"]
                == summary["selected_sha256"] == validation["sha256"], f"Selected hash mismatch: {gpu}")
        require(selected["snapshot"] in paths, f"Selected snapshot absent from manifest: {gpu}")
        require(dict(counts) == summary["counts"], f"Category counts mismatch: {gpu}")
        actual = {str(p.relative_to(directory)) for p in (directory / "kernels").rglob("*.cu")}
        require(actual == paths | {selected["source"]}, f"Unlisted or missing CUDA files: {gpu}")

        grid = validation["source_bound_final_grid"]
        original = verify_reference(grid, records)
        require(original['sources'][grid['source_key']] == selected['sha256'],
                f'Original evidence names another source: {gpu}')
        originals = {(r['phase'], r['batch'], r['length']): r for r in original['rows']}
        workload = config["workload"]
        expected_cells = {(phase, b, length)
                          for phase, lengths in (("prefill", workload["prefill_lengths"]),
                                                 ("decode", workload["decode_context_lengths"]))
                          for b in workload["batches"] for length in lengths}
        cells = {(row["phase"], row["batch"], row["length"]) for row in grid["rows"]}
        require(cells == expected_cells and len(grid["rows"]) == grid["cases"] == 32,
                f"Incomplete selected correctness grid: {gpu}")
        for row in grid["rows"]:
            saved = originals[(row['phase'], row['batch'], row['length'])]
            require(all(saved.get(k) == v for k, v in row.items()), f'Saved grid differs from original evidence: {gpu}')
            require(row["raw_equal"].get(grid["source_key"]) is True, f"Recorded raw-bit failure: {gpu}")
            require(math.isfinite(row['external_max_abs']) and 0 <= row["external_max_abs"]
                    < config["correctness"]["external_final_grid_max_abs_less_than"],
                    f"Recorded external numerical failure: {gpu}")
            for name in (grid['source_key'], row['external_name']):
                value = row['median_ms'][name]
                require(math.isfinite(value) and value > 0, f'Invalid recorded latency: {gpu}')
            if gpu == "h20":
                require(row["changed_input_graph_pass"] is True, "H20 changed-input graph failure")
        for record in validation["expanded_validation"]:
            verify_reference(record, records)
            require(record["all_executed_checks_passed"] is True, f"Expanded validation failure: {gpu}")
        total += len(paths)
        print(f"{gpu.upper()}: {counts['snapshots']} snapshots, {counts['snapshots-demo']} demos; "
              "hashes, selection and 32 saved grid cases verified")
    print(f"PASS: {total} unique source versions across GPU archives, plus three selected copies.")
    verify_figure(directory_root / 'figures', records)
    print(f'PASS: {len(records)} original evidence records; all six figure bars reproduced from 60 timing cells.')
    print("This checks saved evidence and files; it does not compile or execute GPU kernels.")


if __name__ == "__main__":
    main()
