import argparse
import json
import os
from pathlib import Path
import random
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    path = Path(path or ROOT / 'config.yaml').resolve()
    config = yaml.safe_load(os.path.expandvars(path.read_text()))
    config['_config_dir'] = str(path.parent)
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def resolve_path(value, config):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config['_config_dir']) / path).resolve()


def validate(rows, config):
    seen, groups = set(), {}
    for row in rows:
        if row['id'] in seen: raise ValueError(f"Duplicate scene: {row['id']}")
        seen.add(row['id'])
        key = (row['dataset'], row.get('group', row['scene']))
        if key in groups: raise ValueError(f'Duplicate physical scene: {key}')
        groups[key] = row['split']
        if not Path(row['image']).is_file(): raise FileNotFoundError(row['image'])
    for source, spec in config['dataset']['sources'].items():
        for split in ('search', 'final_test'):
            count = sum(r['dataset'] == source and r['split'] == split for r in rows)
            if count != spec[split]:
                raise ValueError(f'{source}/{split}: expected {spec[split]}, found {count}')
    allowed = set(config['dataset']['sources'])
    if any(r['dataset'] not in allowed or r['split'] not in ('search', 'final_test') for r in rows):
        raise ValueError('Unexpected dataset or split')
    return rows


def prepare(config):
    settings = config['dataset']
    manifest = resolve_path(settings['manifest'], config)
    if manifest.exists():
        return validate(json.loads(manifest.read_text()), config)
    rows = []
    for name, spec in settings['sources'].items():
        # An optional frame index selects a reference RGB per physical scene.
        # Entries: {scene, image, group}; relative image paths use the index directory.
        if spec.get('index'):
            index = resolve_path(spec['index'], config)
            entries = json.loads(index.read_text())
            for entry in entries:
                path = Path(entry['image'])
                entry['image'] = str((index.parent / path).resolve())
        else:
            folder = resolve_path(spec['images'], config)
            entries = []
            for scene in sorted(p for p in folder.iterdir() if p.is_dir()):
                images = sorted(scene.glob(spec.get('pattern', '*')))
                images = [p for p in images if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
                if not images: continue
                if len(images) != 1:
                    raise ValueError(f'{scene}: multiple frames; supply a selected-frame index')
                group = scene.name.rsplit('_', 1)[0] if name == 'scannet' else scene.name
                entries.append({'scene': scene.name, 'group': group, 'image': str(images[0])})
        # Select one capture per physical scene before splitting.
        unique = {}
        for entry in sorted(entries, key=lambda e: e['scene']):
            group = entry.get('group') or (entry['scene'].rsplit('_', 1)[0] if name == 'scannet' else entry['scene'])
            entry['group'] = group
            unique.setdefault(group, entry)
        pool = list(unique.values())
        chosen = random.Random(settings['seed']).sample(pool, spec['search'] + spec['final_test'])
        for i, entry in enumerate(chosen):
            rows.append({**entry, 'id': f"{name}/{entry['scene']}", 'dataset': name,
                         'split': 'search' if i < spec['search'] else 'final_test'})
    validate(rows, config)
    write_json(manifest, rows)
    return rows


def public_task(row):
    with Image.open(row['image']) as image:
        width, height = image.size
    # Do not pass scene identity, calibration, depth or labels to the harness.
    return {'image': row['image'], 'width': width, 'height': height}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    args = parser.parse_args()
    rows = prepare(load_config(args.config))
    print({split: sum(r['split'] == split for r in rows) for split in ('search', 'final_test')})
