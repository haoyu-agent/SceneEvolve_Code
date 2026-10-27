import argparse
from functools import lru_cache
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from data import ROOT, load_config, write_json


def quality(ssim, lpips, dino):
    values = np.array([ssim, lpips, dino], dtype=float)
    if not np.isfinite(values).all(): raise ValueError('Non-finite metric')
    return float((np.clip(ssim, 0, 1) + 1 - np.clip(lpips, 0, 1) +
                  (1 + np.clip(dino, -1, 1)) / 2) / 3)


def zero(reason):
    return {'valid': False, 'ssim': 0.0, 'lpips': 1.0, 'dino': -1.0,
            'psnr': 0.0, 'overall': 0.0, 'error': reason}


@lru_cache(maxsize=2)
def metric_models(device, dino_model, lpips_net):
    import lpips
    from transformers import AutoImageProcessor, AutoModel
    perceptual = lpips.LPIPS(net=lpips_net).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(dino_model)
    encoder = AutoModel.from_pretrained(dino_model).eval().to(device)
    return perceptual, processor, encoder


def validate_blend(blend, output_dir, config):
    report = output_dir / 'validated.json'
    report.unlink(missing_ok=True)
    command = [config['blender']['executable'], '--background', '-noaudio', '--factory-startup',
               '--disable-autoexec', '--python-exit-code', '1', '--python', str(ROOT / 'render.py'),
               '--', '--scene', str(blend), '--output-dir', str(output_dir), '--validate-only']
    with (output_dir / 'validation.log').open('w') as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=config['blender']['timeout'])
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0 and report.exists()


def score(reference, output_dir, config):
    output_dir = Path(output_dir)
    try:
        blend, png = output_dir / 'scene.blend', output_dir / 'final.png'
        if not blend.is_file() or not png.is_file():
            return zero('Missing scene.blend or final.png')
        with blend.open('rb') as stream:
            header = stream.read(7)
            if not (header == b'BLENDER' or header.startswith(b'\x28\xb5\x2f\xfd') or header.startswith(b'\x1f\x8b')):
                return zero('Invalid Blender project header')
        with Image.open(reference) as image:
            ref = image.convert('RGB')
            ref.load()
        with Image.open(png) as image:
            pred = image.convert('RGB')
            pred.load()
        if ref.size != pred.size: return zero('Rendered dimensions differ from input')
    except (OSError, ValueError) as error:
        return zero(str(error))
    if not validate_blend(blend, output_dir, config):
        return zero('Blender project cannot be loaded or lacks geometry/camera')
    # Missing weights/dependencies or device failures are infrastructure errors;
    # let them propagate so they cannot silently select a harness on false scores.
    import torch
    from skimage.metrics import structural_similarity
    device = config['metrics']['device']
    perceptual, processor, encoder = metric_models(device, config['metrics']['dino_model'], config['metrics']['lpips_net'])
    a, b = np.asarray(ref), np.asarray(pred)
    small = min(a.shape[:2])
    if small < 3: return zero('Image too small for SSIM')
    win = min(7, small if small % 2 else small - 1)
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=255, win_size=win))
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    # null represents +infinity in JSON for identical images.
    psnr = float(10 * np.log10(255 ** 2 / mse)) if mse else None
    def tensor(image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
    with torch.inference_mode():
        ta, tb = tensor(ref), tensor(pred)
        if min(ta.shape[-2:]) < 64:
            ta = torch.nn.functional.interpolate(ta, size=(64,64), mode='bilinear', align_corners=False)
            tb = torch.nn.functional.interpolate(tb, size=(64,64), mode='bilinear', align_corners=False)
        distance = float(perceptual(ta, tb).item())
        inputs = processor(images=[ref, pred], return_tensors='pt').to(device)
        features = encoder(**inputs).last_hidden_state[:, 0]
        similarity = float(torch.nn.functional.cosine_similarity(features[:1], features[1:]).item())
    return {'valid': True, 'ssim': ssim, 'lpips': distance, 'dino': similarity,
            'psnr': psnr, 'overall': quality(ssim, distance, similarity)}


def aggregate(results):
    if not results: raise ValueError('Cannot aggregate empty results')
    return {'overall': sum(r['overall'] for r in results) / len(results),
            'valid_rate': sum(bool(r['valid']) for r in results) / len(results),
            **{key: sum(r[key] for r in results) / len(results) for key in ('ssim', 'lpips', 'dino')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    write_json(args.output_dir / 'image_metrics.json', score(args.image, args.output_dir, load_config(args.config)))
