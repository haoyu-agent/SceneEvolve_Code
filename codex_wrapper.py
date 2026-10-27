import json
import os
from pathlib import Path
import time
from data import write_json
from processes import run_group


def command(settings, workspace, response, images):
    overrides = {
        'model_provider': 'official_api',
        'model_reasoning_effort': settings['reasoning_effort'],
        'model_providers.official_api.name': 'OpenAI API',
        'model_providers.official_api.base_url': settings['api_base'],
        'model_providers.official_api.env_key': settings['api_key_env'],
        'model_providers.official_api.wire_api': 'responses',
        'model_providers.official_api.requires_openai_auth': False,
    }
    args = [settings['command'], 'exec', '--ignore-user-config', '--ephemeral', '--json',
            '--skip-git-repo-check', '--sandbox', 'workspace-write', '--cd', str(workspace),
            '--model', settings['model'], '--output-last-message', str(response)]
    for key, value in overrides.items(): args.extend(['-c', f'{key}={json.dumps(value)}'])
    for path in images: args.extend(['--image', str(path)])
    args.append('-')
    return args


def run(prompt, config, workspace, images):
    settings = config['proposer']
    if not os.environ.get(settings['api_key_env']): raise ValueError(f"Set {settings['api_key_env']}")
    workspace.mkdir(parents=True, exist_ok=True)
    prompt_path = workspace / 'prompt.txt'
    prompt_path.write_text(prompt)
    response = workspace / 'response.json'
    response.unlink(missing_ok=True)
    started = time.time()
    args = command(settings, workspace, response, images)
    write_json(workspace / 'request.json', {'model': settings['model'], 'images': images, 'command': args})
    with prompt_path.open() as stdin, (workspace / 'events.jsonl').open('w') as stdout, (workspace / 'stderr.log').open('w') as stderr:
        code = run_group(args, timeout=settings['timeout'], stdin=stdin, stdout=stdout, stderr=stderr,
                         cwd=workspace, env=os.environ.copy())
    write_json(workspace / 'session.json', {'started': started, 'seconds': time.time() - started, 'exit_code': code})
    if code: raise RuntimeError(f'Codex proposer exited {code}: {workspace / "stderr.log"}')
    text = response.read_text().strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(text)
