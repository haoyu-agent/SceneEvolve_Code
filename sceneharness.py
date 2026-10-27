import argparse
from datetime import datetime
import json
import shutil
from benchmark import copy_snapshot, evaluate, fixed_digest, freeze, open_run, tree_digest
import codex_wrapper
from data import ROOT, write_json
from evidence import archive_index, candidate_evidence


def read_history(run_dir):
    path = run_dir / 'history.json'
    return json.loads(path.read_text()) if path.exists() else []


def record(run_dir, iteration, summary):
    history = read_history(run_dir)
    if not any(r['agent'] == summary['agent'] for r in history):
        history.append({'iteration': iteration, **{k: v for k, v in summary.items() if k != 'scenes'}})
        write_json(run_dir / 'history.json', history)
    archive_index(run_dir, history)


def update_frontier(run_dir, summary):
    path = run_dir / 'frontier_search.json'
    best = json.loads(path.read_text()) if path.exists() else None
    if best is None or summary['overall'] > best['overall']:
        best = {k: v for k, v in summary.items() if k != 'scenes'}
        write_json(path, best)
    return best


def commit_proposals(workspace, run_dir):
    prepared = json.loads((workspace / 'prepared.json').read_text())
    names = list(prepared)
    for name, digest in prepared.items():
        folder = workspace / name
        if tree_digest(folder) != digest:
            raise RuntimeError(f'Prepared proposal changed: {name}')
        destination = ROOT / 'harnesses' / name
        if destination.exists():
            if tree_digest(destination) != digest:
                raise RuntimeError(f'Existing candidate differs: {destination}')
        else:
            copy_snapshot(folder, destination)
        frozen = freeze(name, run_dir)
        if tree_digest(frozen) != digest:
            raise RuntimeError(f'Frozen candidate differs: {name}')
    write_json(workspace / 'proposals.json', names)
    return names


def proposals(run_dir, config, iteration):
    workspace = run_dir / 'proposals' / f'{iteration:04d}'
    committed = workspace / 'proposals.json'
    if committed.exists(): return json.loads(committed.read_text())
    if (workspace / 'prepared.json').exists(): return commit_proposals(workspace, run_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    best = json.loads((run_dir / 'frontier_search.json').read_text())
    parent = run_dir / 'candidates' / best['agent']
    names = [f'candidate_{run_dir.name}_{iteration:04d}_{j+1}' for j in range(config['evolution']['proposals_per_round'])]
    for name in names:
        folder = workspace / name
        if folder.exists():
            folder.rename(workspace / f'{name}_interrupted_{datetime.now():%Y%m%d%H%M%S%f}')
        shutil.copytree(parent, folder, ignore=shutil.ignore_patterns('__pycache__'))
    history = read_history(run_dir)
    latest = max(r['iteration'] for r in history)
    images = []
    for row in history:
        if row['iteration'] == latest:
            images.extend(json.loads((run_dir / 'evidence' / row['agent'] / 'sheets.json').read_text()))
    prompt = (ROOT / 'prompt-templates/proposer.txt').read_text() + '\n' + json.dumps({
        'round': iteration, 'parent': best, 'candidate_directories': [str(workspace / n) for n in names],
        'archive': str(run_dir / 'evidence' / 'archive.json'),
        'fixed_sceneact_api': str(ROOT / 'sceneact'), 'budget': config['budget'],
        'attached_comparison_sheets': images,
        'required_response': {'candidates': names}}, ensure_ascii=False, indent=2)
    before = fixed_digest()
    snapshot_hashes = {p.name: tree_digest(p) for p in (run_dir / 'candidates').iterdir() if p.is_dir()}
    response = codex_wrapper.run(prompt, config, workspace, images)
    if fixed_digest() != before: raise RuntimeError('Proposer modified fixed runtime')
    if any(tree_digest(run_dir / 'candidates' / name) != digest for name, digest in snapshot_hashes.items()):
        raise RuntimeError('Proposer modified evaluated snapshots')
    if response.get('candidates') != names: raise ValueError(f'Expected exactly {names}')
    for name in names:
        folder = workspace / name
        if any(p.is_symlink() for p in folder.rglob('*')): raise ValueError('Candidate cannot contain symlinks')
        if not (folder / 'policy.py').is_file(): raise ValueError('Candidate needs policy.py')
        for path in folder.rglob('*.py'): compile(path.read_text(), str(path), 'exec')
    write_json(workspace / 'prepared.json', {name: tree_digest(workspace / name) for name in names})
    return commit_proposals(workspace, run_dir)



def finalize(run_dir, config):
    path = run_dir / 'finalized.json'
    if path.exists(): final = json.loads(path.read_text())
    else:
        state = json.loads((run_dir / 'evolution_state.json').read_text())
        if state['completed_rounds'] < config['evolution']['rounds']:
            raise RuntimeError('Finish configured search rounds before held-out evaluation')
        best = json.loads((run_dir / 'frontier_search.json').read_text())
        final = {'agents': {'baseline': 'baseline', 'best': best['agent']}, 'tests': {}, 'status': 'in_progress'}
        write_json(path, final)  # Close search before exposing test evidence.
    for label, name in final['agents'].items():
        if label not in final['tests']:
            summary = evaluate(name, 'final_test', run_dir, config, experiment=f'final_{label}')
            final['tests'][label] = {k: v for k, v in summary.items() if k != 'scenes'}
            write_json(path, final)
    final['status'] = 'complete'
    final['overall_delta'] = final['tests']['best']['overall'] - final['tests']['baseline']['overall']
    write_json(path, final)
    return final


def search(run_dir, config, rounds=None):
    if (run_dir / 'finalized.json').exists(): raise RuntimeError('Search is finalized')
    target = config['evolution']['rounds'] if rounds is None else rounds
    if target < 0 or target > config['evolution']['rounds']: raise ValueError('Invalid target round')
    baseline = evaluate('baseline', 'search', run_dir, config)
    candidate_evidence(baseline, run_dir, config)
    record(run_dir, 0, baseline)
    update_frontier(run_dir, baseline)
    state_file = run_dir / 'evolution_state.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {'completed_rounds': 0}
    for iteration in range(state['completed_rounds'] + 1, target + 1):
        names = proposals(run_dir, config, iteration)
        results = []
        # Both proposals derive from the same retained parent and archived round history.
        for name in names:
            summary = evaluate(name, 'search', run_dir, config)
            candidate_evidence(summary, run_dir, config)
            record(run_dir, iteration, summary)
            results.append(summary)
        winner = max(results, key=lambda r: r['overall'])
        best = update_frontier(run_dir, winner)
        state = {'completed_rounds': iteration, 'best': best['agent']}
        write_json(state_file, state)
        print(f"round={iteration} best={best['agent']} score={best['overall']:.6f}", flush=True)
    if not state_file.exists(): write_json(state_file, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--config')
    parser.add_argument('--rounds', '--iterations', type=int, help='Target completed round, not extra rounds')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    run_dir, config = open_run(args.run_name, args.config)
    if args.test: print(json.dumps(finalize(run_dir, config), indent=2))
    else: search(run_dir, config, args.rounds)


if __name__ == '__main__': main()
