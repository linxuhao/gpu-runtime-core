"""Power limits: an NVIDIA card declaring power_w runs at the lowest power_w of the tenants using it,
its own when none does or while it is lent; the root helper is reached on a unix socket."""
import json
from pathlib import Path
import socket
import sys
import threading
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gpu_runtime
from gpu_runtime import load_config
import harness as base


class Helper:
    """A stand-in for the root helper: records each request and answers ok (or `fail`)."""
    def __init__(self, path):
        self.asked=[];self.fail=None
        self.server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.server.bind(path);self.server.listen(8)
        threading.Thread(target=self.serve,daemon=True).start()
    def serve(self):
        while True:
            try:conn,_=self.server.accept()
            except OSError:return
            with conn:
                ask=json.loads(conn.makefile('r').readline());self.asked.append(ask['watts'])
                reply={'ok':False,'error':self.fail} if self.fail else {'ok':True,'watts':ask['watts']}
                conn.sendall((json.dumps(reply)+'\n').encode())
    def close(self):self.server.close()


class PowerTests(base.SchedulerHarness):
    def setUp(self):
        super().setUp()
        self.helper=Helper(str(self.state/'power.sock'));self.addCleanup(self.helper.close)
        self.s=self.scheduler(base.v2(**{'devices.gpu1.power_w':250,'tenants.sd.device':'gpu1','tenants.sd.api':'cuda','tenants.sd.power_w':220}))
        self.s.power_socket=str(self.state/'power.sock')

    def test_a_tenant_gets_its_limit_while_it_works_and_the_card_its_own_after(self):
        with self.s._using('sd','gpu1'):self.assertEqual(self.helper.asked,[220])
        self.assertEqual(self.helper.asked,[220,250])

    def test_a_tenant_declaring_nothing_runs_at_the_card_limit_and_nothing_is_resent(self):
        with self.s._using('translator','gpu1'):pass
        with self.s._using('translator','gpu1'):pass
        self.assertEqual(self.helper.asked,[250])

    def test_two_users_the_lowest_wins(self):
        with self.s._using('translator','gpu1'):
            with self.s._using('sd','gpu1'):pass
        self.assertEqual(self.helper.asked,[250,220,250])

    def test_a_part_inherits_its_tenant_limit_unless_it_declares_its_own(self):
        self.s.tenants['pair']={'power_w':230,'front':{'power_w':210},'back':{}}
        self.assertEqual((self.s.power_of('pair.front'),self.s.power_of('pair.back'),self.s.power_of('audio')),(210,230,None))

    def test_a_card_without_power_w_and_an_amd_card_are_left_alone(self):
        with self.s._using('audio','gpu0'):pass
        self.s.devices['gpu1'].power_w=None
        with self.s._using('sd','gpu1'):pass
        self.assertEqual(self.helper.asked,[])

    def test_a_refusal_is_logged_the_work_goes_on_and_the_next_use_tries_again(self):
        self.helper.fail='outside 200-250 W'
        with self.s._using('sd','gpu1'):pass
        self.helper.fail=None
        with self.s._using('sd','gpu1'):pass
        self.assertEqual(self.helper.asked,[220,250,220,250])

    def test_no_helper_does_not_stop_the_work(self):
        self.helper.close();(self.state/'power.sock').unlink()
        with self.s._using('sd','gpu1'):ran=True
        self.assertTrue(ran)

    def test_a_lent_card_gets_its_own_limit(self):
        with self.s._using('sd','gpu1'):pass
        self.s.power_sent.clear();self.s.lend('gpu1',{'owner':'t'})
        self.assertEqual(self.helper.asked[-1],250)

    def test_config_checks(self):
        path=self.state/'gpu-config.json'
        for bad in ({'devices.gpu0.power_w':200},{'devices.gpu1.power_w':'250'},{'tenants.sd.power_w':0}):
            path.write_text(json.dumps(base.v2(**bad)))
            with self.assertRaises(ValueError):load_config(path)


if __name__=='__main__':unittest.main()
