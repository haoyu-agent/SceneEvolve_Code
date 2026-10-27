import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from data import write_json


def candidate_evidence(summary, run_dir, config):
    folder = run_dir / 'evidence' / summary['agent']
    folder.mkdir(parents=True, exist_ok=True)
    scenes = []
    for row in summary['scenes']:
        output = Path(row['output_dir'])
        scenes.append({**row, 'steps': [str(p) for p in sorted((output / 'steps').glob('*/result.json'))],
                       'code': str(run_dir / 'candidates' / summary['agent']),
                       'render_history': [str(p) for p in sorted((output / 'steps').glob('*/final.png'))]})
    write_json(folder / 'index.json', {'agent': summary['agent'], 'overall': summary['overall'], 'scenes': scenes})
    size, batch = config['proposer']['contact_sheet_size'], config['proposer']['contact_sheet_rows']
    sheets = []
    for offset in range(0, len(scenes), batch):
        rows = scenes[offset:offset + batch]
        sheet = Image.new('RGB', (2 * size, (size + 38) * len(rows)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows):
            y = i * (size + 38)
            draw.text((4, y + 2), f"{row['id']} score={row['overall']:.4f}", fill='black')
            draw.text((4, y + 18), 'reference', fill='black')
            draw.text((size + 4, y + 18), 'render' if row['render'] else 'MISSING RENDER', fill='black')
            for column, path in enumerate((row['reference'], row['render'])):
                if path and Path(path).exists():
                    with Image.open(path) as image:
                        thumb = ImageOps.contain(image.convert('RGB'), (size, size))
                        sheet.paste(thumb, (column * size + (size - thumb.width)//2, y + 38 + (size - thumb.height)//2))
        path = folder / f'comparison_{offset//batch:03d}.png'
        sheet.save(path)
        sheets.append(str(path))
    write_json(folder / 'sheets.json', sheets)
    return sheets


def archive_index(run_dir, history):
    entries = []
    for row in history:
        entries.append({'iteration': row['iteration'], 'agent': row['agent'], 'overall': row['overall'],
                        'evidence': str(run_dir / 'evidence' / row['agent'] / 'index.json'),
                        'scores': str(run_dir / 'results' / row['agent'] / 'search.json'),
                        'code': str(run_dir / 'candidates' / row['agent'])})
    write_json(run_dir / 'evidence' / 'archive.json', entries)
    return run_dir / 'evidence' / 'archive.json'
