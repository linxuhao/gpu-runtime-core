"""Conflict detection must not need the docker daemon, and must be per device."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from gpu_runtime import Device, Scheduler
from pair_tenant import PAIR
from workers import describe_pid

ROOT=Path(__file__).resolve().parents[1]
CONFIG={'schema':2,'devices':{'gpu0':{'pci':'0000:08:00.0','name':'7900 XTX','vendor':'amd','power_cap_w':280},'gpu1':{'uuid':'GPU-1','vendor':'nvidia'}},
        'tenants':{'pair':PAIR},'policy':{'idle':{'restore':['pair']}}}


class ConflictDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        # The AMD card's sysfs: 0 bytes of VRAM in use, so it is clear whoever reads it.
        sysfs=Path(self.temp.name)/'sysfs';sysfs.mkdir();(sysfs/'mem_info_vram_used').write_text('0')
        p=mock.patch.object(Device,'sysfs',lambda device:sysfs);p.start();self.addCleanup(p.stop)
        self.runtime=Scheduler(ROOT,Path(self.temp.name),CONFIG)
    def tearDown(self):self.temp.cleanup()

    def _run(self, compute=''):
        """Run conflicts() recording every external command it issues."""
        seen=[]
        def record(command,**kwargs):
            seen.append(command[0])
            if command[0]=='nvidia-smi':return compute
            raise AssertionError('conflicts() shelled out to '+command[0])
        with mock.patch('gpu_runtime.subprocess.check_output',record):
            result=self.runtime.conflicts()
        return seen,result

    def test_it_never_consults_the_docker_daemon(self):
        seen,_=self._run()
        self.assertNotIn('docker',seen)

    def test_it_still_asks_nvidia_smi(self):
        # Dropping docker must not quietly drop detection: this is the check that sees
        # compute processes inside other containers, reported as host pids.
        seen,_=self._run()
        self.assertIn('nvidia-smi',seen)

    def test_a_foreign_compute_process_on_our_device_is_reported(self):
        _,result=self._run(compute='4242, GPU-1\n')
        self.assertTrue(any(entry.startswith('pid:4242') for entry in result),result)

    def test_a_process_on_a_card_we_do_not_manage_is_ignored(self):
        _,result=self._run(compute='4242, GPU-other\n')
        self.assertFalse(any('4242' in entry for entry in result),result)

    def test_our_own_workers_are_not_reported(self):
        self.runtime.hold('pair.back','gpu1',mock.Mock(process=mock.Mock(pid=4242)))
        _,result=self._run(compute='4242, GPU-1\n')
        self.assertFalse(any('4242' in entry for entry in result),result)

    def test_a_pid_is_named_with_its_container_when_proc_says_so(self):
        digest='a'*64
        with mock.patch('workers.Path') as path:
            path.return_value.read_text.return_value='0::/system.slice/docker-'+digest+'.scope\n'
            self.assertEqual(describe_pid('77'),'pid:77 (container '+digest[:12]+')')

    def test_a_pid_outside_any_container_is_named_plainly(self):
        with mock.patch('workers.Path') as path:
            path.return_value.read_text.return_value='0::/user.slice/user-1000.slice\n'
            self.assertEqual(describe_pid('77'),'pid:77')

    def test_an_unreadable_proc_entry_degrades_instead_of_raising(self):
        with mock.patch('workers.Path') as path:
            path.return_value.read_text.side_effect=OSError
            self.assertEqual(describe_pid('77'),'pid:77')


if __name__=='__main__':unittest.main()
