"""Exec a resident child that dies if its owning service disappears (Linux)."""
import ctypes
import os
import signal
import sys

parent = os.getppid()
if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0:
    raise RuntimeError("cannot set parent death signal")
if os.getppid() != parent or parent == 1:
    raise SystemExit("parent exited")
os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
