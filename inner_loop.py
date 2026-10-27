import argparse
import json
import importlib.util
import sys
from pathlib import Path
import time
import traceback
from data import load_config, write_json
from runtime import Runtime


def run(task, config, output_dir, source_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(task, config, output_dir)
    started = time.monotonic()
    status = {'error': None}
    try:
        sys.path.insert(0, str(source_dir))
        spec = importlib.util.spec_from_file_location('candidate_policy', source_dir / 'policy.py')
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        policy.solve(task, runtime)
    except Exception:
        status['error'] = traceback.format_exc()
        runtime.log({'event': 'scene_error', 'error': status['error']})
    finally:
        runtime.save_usage()
        write_json(output_dir / 'status.json', {**status, 'seconds': time.monotonic() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    run(json.loads(args.task.read_text()), load_config(args.config), args.output_dir, args.source_dir)
