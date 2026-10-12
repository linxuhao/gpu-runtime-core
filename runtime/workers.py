"""The two process shapes the scheduler places on a device, and how a foreign pid is named.

Both start their child through parent_guard.py (it dies with this service) and inherit the
GPU mode lease fd, so the lease outlives nothing and nothing outlives the lease.
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


class Worker:
    """A JSON-lines worker: prints {"event":"ready"} once, then one result per request."""
    def __init__(self, command, log, lease_fd, env=None):
        self.messages = queue.Queue()
        self.program = os.path.basename(command[0])   # its own messages start "<program>: "
        self.log_path = log
        self.log = open(log, 'a')
        guard = Path(__file__).with_name('parent_guard.py')
        self.process = subprocess.Popen([sys.executable, str(guard), *command], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            pass_fds=(lease_fd,), env=env)
        self.reader=threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.ready = self.receive('ready', 240)

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and (obj.get('event') in ('ready','result') or obj.get('status') == 'completed'):
                        self.messages.put(obj)
                        continue
                except ValueError:
                    pass
                self.log.write(line); self.log.flush()
        finally:
            self.messages.put(None)

    def receive(self, expected, timeout=600):
        try:
            value = self.messages.get(timeout=timeout)
        except queue.Empty:
            self.stop()
            raise RuntimeError('worker timeout; see preserved log')
        if value is None or (expected == 'ready' and value.get('event') != 'ready'):
            self.stop()
            raise RuntimeError('worker exited without expected response: %s; see %s' % (self.last_error(), self.log_path))
        return value

    def last_error(self):
        """The worker's own last word: the final exception line of its log, e.g. a CUDA OOM."""
        try:lines = [l.strip() for l in open(self.log_path, errors='replace').read()[-200000:].splitlines() if l.strip()]
        except OSError:return 'no log'
        errors = [l for l in lines if ('Error' in l or 'error' in l or l.startswith(self.program+':')) and not l.startswith('File ')]
        return (errors or lines or ['empty log'])[-1][:500]

    def run(self, request, timeout=600):
        self.process.stdin.write(json.dumps(request)+'\n'); self.process.stdin.flush()
        return self.receive('result', timeout)

    def stop(self):
        if self.log.closed:return
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        self.reader.join(timeout=5)
        if self.reader.is_alive():raise RuntimeError('worker stdout did not close')
        self.process.stdin.close();self.process.stdout.close();self.log.close()


class ServerWorker:
    """An HTTP engine: started, polled on its health route until it answers, stopped on demand."""
    def __init__(self, command, log, lease_fd, env, cwd, url, health='/health', startup_s=240):
        self.log = open(log, 'ab')
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name('parent_guard.py')), *command],
            stdout=self.log, stderr=self.log, pass_fds=(lease_fd,), env=env, cwd=cwd)
        self.url = url
        try:
            deadline = time.monotonic() + startup_s
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('engine exited; see '+str(log))
                try:
                    with urllib.request.urlopen(url+health, timeout=2) as response:
                        if response.status == 200:return
                except (OSError, urllib.error.URLError):pass
                time.sleep(.2)
            raise RuntimeError('engine startup timeout; see '+str(log))
        except BaseException:
            self.stop();raise

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()


def describe_pid(pid):
    """Name a conflicting process, including its container, without the docker daemon.

    A containerised process carries its container id in its own cgroup path, so /proc
    answers what `docker inspect` used to — with no socket and no daemon privilege."""
    try:cgroup = Path('/proc',pid,'cgroup').read_text()
    except OSError:return 'pid:'+pid
    for token in cgroup.replace('/',' ').replace('-',' ').replace('.',' ').split():
        if len(token)==64 and all(c in '0123456789abcdef' for c in token):
            return 'pid:'+pid+' (container '+token[:12]+')'
    return 'pid:'+pid
