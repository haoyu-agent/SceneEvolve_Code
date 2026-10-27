import base64
import copy
import mimetypes
import os
from pathlib import Path
import time
from openai import OpenAI


def read_key(settings):
    return os.environ[settings['api_key_env']]


def image_message(path):
    return {'type': 'image_file', 'path': str(Path(path).resolve())}


def encode_messages(messages):
    result = copy.deepcopy(messages)
    for message in result:
        if isinstance(message.get('content'), list):
            for i, part in enumerate(message['content']):
                if part['type'] == 'image_file':
                    path = Path(part['path'])
                    mime = mimetypes.guess_type(path.name)[0] or 'image/png'
                    data = base64.b64encode(path.read_bytes()).decode()
                    message['content'][i] = {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{data}'}}
    return result


class ImageLLM:
    def __init__(self, settings, log, deadline):
        self.settings, self.log, self.deadline = settings, log, deadline
        self.client = OpenAI(api_key=read_key(settings), base_url=settings['api_base'], max_retries=0)
        self.usage = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'seconds': 0.0}

    def __call__(self, messages):
        timeout = min(self.settings['timeout'], self.deadline - time.monotonic())
        if timeout <= 0: raise TimeoutError('Scene deadline reached')
        self.usage['calls'] += 1
        self.log({'event': 'model_request', 'messages': messages, 'model': self.settings['name'],
                  'reasoning_effort': self.settings['reasoning_effort']})
        started = time.monotonic()
        text, reasoning = '', ''
        try:
            stream = self.client.chat.completions.create(
                model=self.settings['name'], reasoning_effort=self.settings['reasoning_effort'],
                messages=encode_messages(messages), timeout=timeout, stream=True,
                stream_options={'include_usage': True}, **self.settings.get('parameters', {}))
            with stream:
                for chunk in stream:
                    raw = chunk.model_dump(exclude_none=True)
                    self.log({'event': 'model_chunk', 'chunk': raw})
                    if chunk.usage:
                        self.usage['input_tokens'] += chunk.usage.prompt_tokens
                        self.usage['output_tokens'] += chunk.usage.completion_tokens
                    if time.monotonic() >= self.deadline: raise TimeoutError('Scene deadline reached')
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        text += delta.content or ''
                        reasoning += getattr(delta, 'reasoning_content', '') or ''
            message = {'role': 'assistant', 'content': text}
            if reasoning: message['reasoning_content'] = reasoning
            self.log({'event': 'model_response', 'message': message})
            return message
        except Exception as error:
            self.log({'event': 'model_error', 'error': repr(error), 'partial_content': text,
                      'partial_reasoning': reasoning})
            raise
        finally:
            self.usage['seconds'] += time.monotonic() - started
