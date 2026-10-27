import argparse
import json
from pathlib import Path
import runpy
import sys


def render_scene(output_dir, settings):
    import bpy
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError('Scene needs a camera matching the input photograph')
    if not any(o.type == 'MESH' and len(o.data.polygons) for o in scene.objects):
        raise ValueError('Scene has no mesh geometry')
    scene.render.engine = settings['engine']
    scene.render.resolution_x, scene.render.resolution_y = settings['width'], settings['height']
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = settings['samples']
        scene.cycles.seed = settings['seed']
        scene.cycles.device = settings['device']
        scene.cycles.use_adaptive_sampling = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.filepath = str(output_dir / 'final.png')
    # Save and render the identical scene. No calibration or forced camera pose.
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / 'scene.blend'))
    bpy.ops.render.render(write_still=True)
    (output_dir / 'camera.json').write_text(json.dumps({
        'matrix_world': [list(row) for row in scene.camera.matrix_world],
        'lens': scene.camera.data.lens, 'type': scene.camera.data.type,
        'width': settings['width'], 'height': settings['height']}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.code:
        runpy.run_path(str(args.code), run_name='__sceneact_build__')
    else:
        import bpy
        bpy.ops.wm.open_mainfile(filepath=str(args.scene))
    if args.validate_only:
        import bpy
        scene = bpy.context.scene
        if scene.camera is None or not any(o.type == 'MESH' and len(o.data.polygons) for o in scene.objects):
            raise ValueError('Submission needs mesh geometry and an active camera')
        (args.output_dir / 'validated.json').write_text(json.dumps({'valid': True}))
    else:
        render_scene(args.output_dir, json.loads(args.settings.read_text()))


if __name__ == '__main__':
    import os
    import traceback
    try:
        main()
        status = 0
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
