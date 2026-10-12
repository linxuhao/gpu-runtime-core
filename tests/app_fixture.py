"""A minimal application on the job service, for the tests: `render` submits a job of kind render,
which the fixture runtimes run as two stages (encode, then sample) on two devices, and `pair` one
of the pair_tenant plugin's kind."""
import job_service


class Service(job_service.Service):
    PREFIXES={**job_service.Service.PREFIXES,'render':'take','pair':'pair'}

    def call(self, action, args):
        if action in ('render','pair'):
            with self.cv:
                if self.draining:raise ValueError('GPU resources released; acquire before submitting')
                if not isinstance(args.get('request'),dict):raise ValueError('request must be an object')
                req=dict(args['request']);origin=req.get('origin',{})
                return self._enqueue(str(origin.get('project','')),origin.get('shot'),action,req,args['request_key'])
        return super().call(action,args)
