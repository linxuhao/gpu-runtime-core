"""The image's Vulkan userspace against the host's. The runtime image pins Mesa to the host's
exact versions because the RADV build sets the engines' numerics; a host `apt upgrade`
moves the host alone, and engines then compute differently from what was measured on the host. This
compares the two dpkg databases and says what differs, for resource_status and gpu-top.

The host's database is mounted read-only as a directory: dpkg replaces `status` by rename, so a
file mount would keep showing the version before an upgrade.
"""
import os
from pathlib import Path
import sys
import threading

PACKAGES=('mesa-vulkan-drivers','mesa-libgallium','libegl-mesa0','libvulkan1')
HOST_STATUS='/host/dpkg/status'
OWN_STATUS='/var/lib/dpkg/status'


def installed(text, packages=PACKAGES):
    """package -> version of those installed for amd64 (or all), from a dpkg status file."""
    found={}
    for block in text.split('\n\n'):
        fields=dict(line.split(': ',1) for line in block.splitlines() if ': ' in line and not line.startswith(' '))
        if fields.get('Package') in packages and fields.get('Status','').endswith(' installed') and fields.get('Architecture') in ('amd64','all'):
            found[fields['Package']]=fields.get('Version')
    return found


def drift(host=HOST_STATUS, own=OWN_STATUS):
    """One line per package installed on both whose versions differ; None when the host's database
    is not there (not in the container, or not mounted)."""
    try:theirs=installed(Path(host).read_text())
    except OSError:return None
    mine=installed(Path(own).read_text())
    return ['%s: host %s, image %s'%(p,theirs[p],mine[p]) for p in PACKAGES if p in theirs and p in mine and theirs[p]!=mine[p]]


class MesaCheck:
    """drift(), read again only when the host's database changed."""
    def __init__(self, host=None, own=OWN_STATUS):
        self.host=host or os.environ.get('HOST_DPKG_STATUS',HOST_STATUS);self.own=own
        self.lock=threading.Lock();self.stamp=None;self.found=[]

    def warnings(self):
        try:stamp=os.stat(self.host).st_mtime_ns
        except OSError:return []
        with self.lock:
            if stamp!=self.stamp:
                self.stamp=stamp;differs=drift(self.host,self.own) or []
                self.found=['Vulkan userspace differs from the host, so Vulkan engines do not compute as on the host; '
                            'pin the image\'s Mesa packages to the host versions and rebuild: '+'; '.join(differs)] if differs else []
                for line in self.found:print('WARNING '+line,file=sys.stderr,flush=True)
            return list(self.found)
