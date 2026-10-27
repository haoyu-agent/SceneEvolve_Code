import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import yaml
from data import ROOT, load_config, prepare, public_task, write_json
from evaluate import aggregate
from processes import run_group


def tree_digest(folder):
    result = hashlib.sha256()
    for path in sorted(Path(folder).rglob('*')):
        if (path.is_file() and '__pycache__' not in path.parts
                and not any(part.endswith('.bak') or '.bak.' in part for part in path.relative_to(folder).parts)):
            result.update(str(path.relative_to(folder)).encode())
            result.update(path.read_bytes())
    return result.hexdigest()


def open_run(run_name, config_path=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_name): raise ValueError('Invalid run name')
    run_dir = ROOT / 'logs' / run_name
    config_file = run_dir / 'config.yaml'
    if config_file.exists():
        config = load_config(config_file)
        recorded = json.loads((run_dir / 'fixed_code.json').read_text())
        if fixed_digest() != recorded: raise RuntimeError('Fixed runtime changed; create a new run')
        return run_dir, config
    config = load_config(config_path)
    rows = prepare(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'splits.json', rows)
    config['dataset']['manifest'] = str(run_dir / 'splits.json')
    config.pop('_config_dir', None)
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(run_dir / 'fixed_code.json', fixed_digest())
    return run_dir, load_config(config_file)


def fixed_digest():
    paths = ['sceneact', 'runtime.py', 'llm.py', 'render.py', 'evaluate.py',
             'inner_loop.py', 'benchmark.py', 'data.py', 'processes.py', 'sceneharness.py',
             'codex_wrapper.py', 'evidence.py', 'prompt-templates']
    return {p: tree_digest(ROOT / p) if (ROOT / p).is_dir() else hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def copy_snapshot(source, target):
    temporary = target.with_name('.' + target.name + '.copying')
    if temporary.exists(): shutil.rmtree(temporary)
    shutil.copytree(source, temporary, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.bak', '*.bak.*'))
    temporary.replace(target)


def freeze(name, run_dir):
    target = run_dir / 'candidates' / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_snapshot(ROOT / 'harnesses' / name, target)
    return target


def evaluate(name, split, run_dir, config, experiment=None):
    source = freeze(name, run_dir)
    code_hash = tree_digest(source)
    rows = [r for r in json.loads((run_dir / 'splits.json').read_text()) if r['split'] == split]
    if not rows: raise ValueError('Empty split')
    label = experiment or name
    config_path = run_dir / 'config.yaml'
    def execute(row):
        # Opaque scene directory keeps dataset identifiers out of student prompts.
        scene_key = hashlib.sha256(row['id'].encode()).hexdigest()[:16]
        output = ROOT / 'jobs' / run_dir.name / label / scene_key
        result_path = output / 'metrics.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['code_hash'] != code_hash: raise RuntimeError('Candidate changed after evaluation')
            return result
        output.mkdir(parents=True, exist_ok=True)
        # Interrupted, unscored attempts are retained but never reused as successful submissions.
        if (output / 'task.json').exists() and not (output / 'generation_complete.json').exists():
            archived = output.with_name(output.name + f'_interrupted_{time.time_ns()}')
            output.rename(archived)
            output.mkdir()
        task = public_task(row)
        reference = output / 'reference.png'
        from PIL import Image
        with Image.open(task['image']) as image: image.convert('RGB').save(reference)
        task['image'] = str(reference)
        write_json(output / 'task.json', task)
        # Candidate process receives only public task, runtime settings and a candidate snapshot.
        runtime_config = {k: config[k] for k in ('model', 'budget', 'render', 'blender')}
        write_json(output / 'runtime.json', runtime_config)
        started = time.monotonic()
        if not (output / 'generation_complete.json').exists():
            error = None
            with (output / 'worker_stdout.log').open('w') as stdout, (output / 'worker_stderr.log').open('w') as stderr:
                try:
                    code = run_group([sys.executable, str(ROOT / 'inner_loop.py'), '--config', str(output / 'runtime.json'),
                        '--task', str(output / 'task.json'), '--source-dir', str(source), '--output-dir', str(output)],
                        timeout=config['budget']['seconds'], cwd=output, stdout=stdout, stderr=stderr)
                    if code: error = f'Worker exited {code}'
                except subprocess.TimeoutExpired: error = 'Scene wall-clock timeout'
            write_json(output / 'generation_complete.json', {'seconds': time.monotonic() - started, 'error': error})
        generation = json.loads((output / 'generation_complete.json').read_text())
        # A completed previous build remains eligible after a later generation failure.
        with (output / 'scoring_stdout.log').open('w') as stdout, (output / 'scoring_stderr.log').open('w') as stderr:
            code = run_group([sys.executable, str(ROOT / 'evaluate.py'), '--config', str(config_path),
                '--image', str(reference), '--output-dir', str(output)], timeout=config['metrics']['timeout'],
                cwd=ROOT, stdout=stdout, stderr=stderr)
        if code: raise RuntimeError(f'Scorer failed: {output / "scoring_stderr.log"}')
        metrics = json.loads((output / 'image_metrics.json').read_text())
        usage_path = output / 'usage.json'
        usage = json.loads(usage_path.read_text()) if usage_path.exists() else {
            'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'seconds': 0, 'executions': 0}
        result = {**metrics, 'id': row['id'], 'dataset': row['dataset'], 'split': split,
            'candidate': name, 'code_hash': code_hash, 'usage': usage, 'seconds': generation['seconds'],
            'generation_error': generation['error'], 'output_dir': str(output),
            'reference': str(reference), 'render': str(output / 'final.png') if (output / 'final.png').exists() else None,
            'trajectory': str(output / 'trajectory.jsonl')}
        write_json(result_path, result)
        return result
    with ThreadPoolExecutor(max_workers=config['benchmark']['concurrency']) as pool:
        results = list(pool.map(execute, rows))
    if tree_digest(source) != code_hash: raise RuntimeError('Candidate snapshot mutated during execution')
    summary = {'agent': name, 'split': split, 'count': len(results), **aggregate(results),
        'datasets': {source: aggregate([r for r in results if r['dataset'] == source]) for source in sorted({r['dataset'] for r in results})},
        'usage': {key: sum(r['usage'][key] for r in results) for key in ('calls', 'input_tokens', 'output_tokens', 'seconds', 'executions')},
        'scene_seconds': sum(r['seconds'] for r in results), 'scenes': results}
    write_json(run_dir / 'results' / label / f'{split}.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--config')
    parser.add_argument('--agent', default='baseline')
    args = parser.parse_args()
    run_dir, config = open_run(args.run_name, args.config)
    if (run_dir / 'finalized.json').exists(): raise RuntimeError('Search closed')
    print(json.dumps(evaluate(args.agent, 'search', run_dir, config), indent=2))
