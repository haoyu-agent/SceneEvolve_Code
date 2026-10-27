import os
import signal
import subprocess


def run_group(command, *, timeout, **kwargs):
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            process.communicate(timeout=timeout)
        except BaseException:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
            raise
        finally:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        return process.returncode
