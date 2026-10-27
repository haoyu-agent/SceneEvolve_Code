import json
from pathlib import Path
import re


def solve(task, runtime):
    prompt = (Path(__file__).parent / 'system.txt').read_text()
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': [
        {'type': 'text', 'text': 'Reconstruct this RGB image. Match its camera and framing. ' +
         f"Image dimensions: {task['width']} x {task['height']}. Budget: {runtime.remaining()}"},
        runtime.image(task['image'])]}]
    while runtime.remaining()['model_calls'] > 0 and runtime.remaining()['seconds'] > 0:
        response = runtime.ask(messages)
        messages.append(response)
        text = response['content'] or ''
        blocks = re.findall(r'```python\s*\n(.*?)```', text, re.S)
        if not blocks:
            if text.strip().upper() == 'DONE': return
            messages.append({'role': 'user', 'content': 'Return one complete program in a python code block, or DONE.'})
            continue
        if runtime.remaining()['executions'] <= 0: return
        result = runtime.execute(blocks[-1])
        feedback = {'type': 'text', 'text': json.dumps({k: v for k, v in result.items() if k != 'image'}) +
                    '\nCompare the rendered camera view with the original reference. ' +
                    f'Fix errors with a complete revised program, or finish with DONE. Remaining: {runtime.remaining()}'}
        content = [feedback]
        if result['image']: content.append(runtime.image(result['image']))
        messages.append({'role': 'user', 'content': content})
