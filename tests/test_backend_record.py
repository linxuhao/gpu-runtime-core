"""backend_record: the runtime checks each server process it starts, so that a reply can be tied
to the process that produced it (X-GPU-Backend)."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import backend_record
from gpu_runtime import descendants


class BackendRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.recorder=backend_record.Recorder(self.root/'backends',descendants)
    def tearDown(self):self.temp.cleanup()

    def child(self):
        process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],env={**os.environ,'GGML_VK_VISIBLE_DEVICES':'1','SOME_SECRET':'x'})
        self.addCleanup(lambda:(process.kill(),process.wait()));return process

    def test_a_server_of_ours_is_checked_and_named_by_its_process(self):
        process=self.child();name=self.recorder.started('qwen3-q4','gpu1','7900 XTX',process.pid,wait=True)
        self.assertRegex(name,r'^[0-9a-f]{8}-%d-\d+$'%process.pid)
        record=self.recorder.get(name)
        self.assertEqual((record['status'],record['tenant'],record['device'],record['card'],record['alive']),('checked','qwen3-q4','gpu1','7900 XTX',True))
        exe=os.readlink('/proc/%d/exe'%process.pid)
        self.assertEqual(record['exe'],{'path':exe,'sha256':hashlib.sha256(Path(exe).read_bytes()).hexdigest()})
        self.assertEqual(record['argv'][1:],['-c','import time;time.sleep(60)'])
        self.assertEqual(record['env'].get('GGML_VK_VISIBLE_DEVICES'),'1');self.assertNotIn('SOME_SECRET',record['env'],'only what sets numerics')
        process.kill();process.wait()
        self.assertFalse(self.recorder.get(name)['alive'],'a reload is another process, another id')

    def test_only_our_own_children_are_looked_inside(self):
        self.assertIsNone(self.recorder.started('x','gpu0','card',1))
        self.assertIsNone(self.recorder.started('x','gpu0','card',2**22+12345),'no such process')

    def test_unknown_ids_are_refused(self):
        for name in ('../../etc/passwd','abc','0123abcd-1-2'):
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'unknown backend'):self.recorder.get(name)

    def test_the_weights_a_text_server_was_given(self):
        llama=['/b/llama-server','-m','/w/q4.gguf','-md','/w/mtp.gguf','--mmproj','/w/mm.gguf','-c','131072']
        self.assertEqual(backend_record.weights(llama),['/w/q4.gguf','/w/mtp.gguf','/w/mm.gguf'])
        folder=self.root/'Qwen3.8-27B-FP8';folder.mkdir()
        for name in ('model-00001.safetensors','config.json','tokenizer.json','README.md'):(folder/name).write_text('x')
        self.assertEqual([Path(p).name for p in backend_record.weights(['/v/bin/python','-m','vllm.entrypoints.cli.main','serve',str(folder)])],
                         ['config.json','model-00001.safetensors','tokenizer.json'])
        self.assertEqual(backend_record.weights(['/x/audiocpp_server','-m','/w/a.gguf']),[],'not a text server')

    def test_a_file_is_hashed_once_until_it_changes(self):
        path=self.root/'w.gguf';path.write_bytes(b'one');stat=path.stat()
        first=self.recorder.hashes(str(path))
        path.write_bytes(b'two');os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        self.assertEqual(backend_record.Hashes(self.root/'backends'/'sha256-cache.json')(str(path)),first,'same size and mtime: from the cache, also after a restart')
        os.utime(path,ns=(1,stat.st_mtime_ns+1))
        self.assertEqual(self.recorder.hashes(str(path)),hashlib.sha256(b'two').hexdigest())


if __name__=='__main__':unittest.main()
