"""A plugin tenant for the tests (gpu-config "kind": "pair"): two parts on two devices, like a model
whose encoder runs on one card while its main network stays resident on another. A `pair` job is a
`first` stage on the front part's device (or its fallback), then a `second` on the back part's."""
from gpu_runtime import TenantKind

PAIR={'kind':'pair','front':{'device':'gpu0','api':'vulkan'},'back':{'device':'gpu1','api':'cuda'}}


class Tenant(TenantKind):
    jobs=('pair',)

    @staticmethod
    def parts(spec):return {'front':spec['front'],'back':spec['back']}

    @staticmethod
    def validate(name, spec, devices):
        if spec['back'].get('fallback') is not None:raise ValueError('tenant '+name+': back.fallback: the back part has no fallback')

    def part(self, part):return self.scheduler.part(self.name+'.'+part)
    def stages(self, kind, request):return [('first',self.scheduler.lane_for(self.spec['front'])),('second',self.spec['back']['device'])]
    def first_slot(self, kind, request):return self.spec['front']
    def part_of(self, kind, stage):return self.name+('.front' if stage=='first' else '.back')
    def home(self):return self.spec['back']['device']
    def restore(self):self.start('back')

    def resident(self, device):
        if self.part('front') and self.scheduler.located(self.name+'.front')==device:return 'first'
        if self.part('back') and self.spec['back']['device']==device:return 'second'
        return None

    def start(self, part, device=None):
        """The part's worker on `device` (default: where its work goes now), started there if need be."""
        s=self.scheduler;name=self.name+'.'+part;slot=self.spec[part]
        device=s.devices[device or s.lane_for(slot)]
        with s.device_locks[device.name]:
            s._lease()
            if self.part(part) and s.located(name)==device.name:return self.part(part)
            s._usable(device);device.check()
            s._leave(name,device.name)
            try:
                s._place(name,device)
                with s._phase():worker=s.worker([part],s._log(part),s.lease.fileno(),device.env(slot['api']))
                s.hold(name,device.name,worker)
            finally:s._started(name)
            s._refresh()
            return worker

    def run(self, kind, stage, request, out, lane):
        part='front' if stage=='first' else 'back';device=lane if part=='front' else self.spec['back']['device']
        s=self.scheduler
        with s.device_locks[device],s._using(self.name+'.'+part,device):
            return self.start(part,device).run(request)

    def status(self):
        return {part+'_device':self.scheduler.located(self.name+'.'+part) for part in ('front','back')}
