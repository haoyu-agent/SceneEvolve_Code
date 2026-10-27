import json
from pathlib import Path
import shutil
import subprocess
import time
from PIL import Image
from data import ROOT, write_json
from llm import ImageLLM, image_message


class BudgetExceeded(RuntimeError):
    pass


class Runtime:
    def __init__(self, task, config, output_dir):
        self.task, self.config = task, config
        self.output_dir = Path(output_dir)
        self.deadline = time.monotonic() + config['budget']['seconds']
        self.calls, self.executions = 0, 0
        self.llm = None
        self.save_usage()

    def log(self, event):
        with (self.output_dir / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), **event}, ensure_ascii=False) + '\n')

    def remaining(self):
        return {'model_calls': self.config['budget']['model_calls'] - self.calls,
                'executions': self.config['budget']['executions'] - self.executions,
                'seconds': max(0, self.deadline - time.monotonic())}

    def check(self, resource):
        remaining = self.remaining()
        if remaining['seconds'] <= 0 or remaining[resource] <= 0:
            raise BudgetExceeded(f'Budget exhausted: {remaining}')

    def save_usage(self):
        usage = self.llm.usage if self.llm else {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'seconds': 0.0}
        write_json(self.output_dir / 'usage.json', {**usage, 'calls': self.calls, 'executions': self.executions})

    def ask(self, messages):
        self.check('model_calls')
        self.calls += 1
        try:
            if self.llm is None:
                self.llm = ImageLLM(self.config['model'], self.log, self.deadline)
            return self.llm(messages)
        finally:
            self.save_usage()

    def execute(self, code):
        """Execute a complete scene program and render it; one budget unit, even on failure."""
        self.check('executions')
        self.executions += 1
        self.save_usage()
        folder = self.output_dir / 'steps' / f'{self.executions:03d}'
        folder.mkdir(parents=True)
        (folder / 'build.py').write_text(code)
        settings = {**self.config['render'], 'width': self.task['width'], 'height': self.task['height']}
        write_json(folder / 'render_settings.json', settings)
        command = [self.config['blender']['executable'], '--background', '-noaudio', '--factory-startup',
                   '--python-exit-code', '1', '--python', str(ROOT / 'render.py'), '--',
                   '--code', str(folder / 'build.py'), '--settings', str(folder / 'render_settings.json'),
                   '--output-dir', str(folder)]
        self.log({'event': 'execute', 'step': self.executions, 'code': code, 'command': command})
        error = None
        try:
            with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
                subprocess.run(command, cwd=folder, stdout=stdout, stderr=stderr, check=True,
                               timeout=min(self.config['blender']['timeout'], self.remaining()['seconds']))
            with Image.open(folder / 'final.png') as image:
                image.load()
                if image.size != (self.task['width'], self.task['height']):
                    raise ValueError('Render dimensions do not match input')
            if not (folder / 'scene.blend').stat().st_size: raise ValueError('Empty Blender project')
        except Exception as exc:
            error = repr(exc)
        result = {'ok': error is None, 'step': self.executions, 'error': error,
                  'stdout': (folder / 'stdout.log').read_text(errors='replace') if (folder / 'stdout.log').exists() else '',
                  'stderr': (folder / 'stderr.log').read_text(errors='replace') if (folder / 'stderr.log').exists() else '',
                  'image': str(folder / 'final.png') if error is None else None}
        self.log({'event': 'execution_result', **result})
        write_json(folder / 'result.json', result)
        if result['ok']: self.select(self.executions)
        return result

    def select(self, step):
        folder = self.output_dir / 'steps' / f'{step:03d}'
        if not json.loads((folder / 'result.json').read_text())['ok']:
            raise ValueError('Cannot submit a failed step')
        for name in ('scene.blend', 'final.png', 'build.py', 'camera.json'):
            shutil.copyfile(folder / name, self.output_dir / name)
        write_json(self.output_dir / 'selected.json', {'step': step})
        self.log({'event': 'select', 'step': step})

    def image(self, path):
        return image_message(path)
