# gpu-runtime

A declarative runtime for the GPUs of one Linux host. You describe the cards and what runs on them
in one JSON file; the runtime starts and stops the processes, keeps a durable queue per card, and
serves them over an OpenAI-compatible HTTP API by model name.

It is meant for a private deployment: a workstation or a server with a few GPUs shared by people,
agents and long-running research, where several models do not fit at once and somebody has to
decide what is loaded where. Swapping a card, adding a model or moving one to another card is an
edit of the config, not of the code.

- **Placement.** A card is `exclusive` (whatever is placed evicts the rest) or `lru` (tenants stay
  side by side while the card has room for what the newcomer declares, measured; otherwise the
  least recently used leave first, visitors before the card's idle residents).
- **One queue per card**, durable (SQLite): jobs survive a restart or are marked interrupted, never
  silently lost. A job that needs two cards is two stages, so no card is held while it waits for the
  other.
- **Fallback and spill.** A tenant may declare a second card: its work goes there while its own card
  is lent, blocked or drained, and with `spill` also while its own card is busy with someone else.
  A call may restrict itself to some cards (`devices`) when outputs must be bit-identical.
- **Lending.** A card can be lent to a person or a script (`gpu_borrow.py`): queued among the jobs,
  or exclusively for days; nothing of the runtime touches it until it is returned.
- **Foreign processes.** A process on a card that the runtime did not start blocks that card; it is
  never killed.
- **Provenance.** Every reply names the device (`X-GPU-Device`) and the process (`X-GPU-Backend`)
  that produced it; `GET /v1/backends/<id>` returns that process's executable, command line,
  numerics-relevant environment, GPU libraries and weights, each with its sha256, as the runtime
  checked them.
- **Plugins.** Anything beyond an HTTP server or a batch script is a tenant kind written in Python
  (`TenantKind`): parts placed on their own cards, jobs made of stages.

Cards: NVIDIA (through `nvidia-smi`, CUDA or Vulkan) and AMD (through sysfs, Vulkan with Mesa RADV).
The runtime itself is Python 3 standard library only; the HTTP facade needs `starlette`, `uvicorn`
and `httpx`.

## Quick start

```bash
mkdir -p state
cp examples/gpu-config.json state/        # then edit: your card's UUID (nvidia-smi -L), your models
head -c 32 /dev/urandom | base64 > state/gpu-api.token
python3 runtime/job_service.py --state state &          # the runtime, on state/runtime.sock
pip install -r facade/requirements.txt
GPU_STATE=state GPU_TOKEN_FILE=state/gpu-api.token python3 facade/app.py --port 9041 &
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9041/v1", api_key=open("state/gpu-api.token").read().strip())
reply = client.chat.completions.with_raw_response.create(model="qwen3-8b", messages=[{"role": "user", "content": "hi"}])
print(reply.headers["x-gpu-device"], reply.parse().choices[0].message.content)
```

`examples/docker-compose.yml` runs the same two processes in containers. `runtime/gpu_top.py` shows
the cards, their queues and the text servers' prefill and decode speeds, as the engines time them
(vLLM's `/metrics`; llama-server's `/metrics`, so start it with `--metrics`):

```bash
python3 runtime/gpu_top.py -w
```

## Configuration: `state/gpu-config.json`

```json
{
  "schema": 2,
  "devices": {
    "gpu0": {"vendor": "nvidia", "uuid": "GPU-…", "name": "RTX 4090", "placement": "lru", "queue": "affinity"},
    "gpu1": {"vendor": "amd", "pci": "0000:08:00.0", "name": "7900 XTX", "power_cap_w": 280}
  },
  "tenants": {
    "asr":     {"kind": "server", "device": "gpu1", "fallback": "gpu0", "api": "vulkan", "vram_gib": 3,
                "command": ["…"], "url": "http://127.0.0.1:19000", "routes": ["/health", "/v1/audio/transcriptions"]},
    "upscale": {"kind": "batch", "device": "gpu0", "api": "cuda", "python": "/opt/venv/bin/python",
                "script": "/opt/upscale/run.py", "settings": {"model": "/models/x4"}, "timeout_s": 3600}
  },
  "services": {
    "chat": {"names": ["qwen3-8b"], "slots": 4,
             "backends": [{"device": "gpu1", "api": "vulkan", "command": ["…"], "url": "http://127.0.0.1:18000"},
                          {"device": "gpu0", "api": "cuda",   "command": ["…"], "url": "http://127.0.0.1:18000"}]}
  },
  "policy": {"idle": {"restore": ["chat", "asr"]}}
}
```

### devices

| key | |
|---|---|
| `vendor` | `nvidia` (addressed by `uuid`) or `amd` (addressed by `pci`) |
| `name` | what it is called on screen |
| `placement` | `exclusive` (default) or `lru` |
| `queue` | `priority` (default: calls by tenant name first, then by age), `fifo`, or `affinity` (also prefers jobs whose tenant is loaded, each passed over a bounded number of times) |
| `vram_margin_gib` | kept free on an `lru` card beyond what the newcomer declares (default 0.5) |
| `power_cap_w` | AMD: the card's hwmon power cap must be at or below this before anything starts |
| `vulkan_icd`, `vulkan_icd_path`, `vulkan_index` | which Vulkan driver manifest and which device of it a Vulkan process gets |

### tenants

A **server** (`"kind": "server"`, the default) is an HTTP server the runtime starts and calls:

| key | |
|---|---|
| `device`, `api` | its card, and `vulkan` or `cuda`: they set the environment that lands it there |
| `command`, `cwd`, `env`, `url` | how it is started and reached |
| `health`, `startup_s` | the route polled until it answers (default `/health`), and for how long (default 240 s) |
| `vram_gib` | what it needs on an `lru` card; without it, it is placed as on an exclusive card |
| `routes` | the paths callers may use (default: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) |
| `poll` | for a server that answers a submission with a job to poll: `{"submit", "field", "prefix"}`; the facade answers at once with a job of its own to poll, and the runtime polls the server |
| `models` | the model names it answers to on `/v1` |
| `slots` | how many calls it takes at once (default 1) |
| `linger_s` | how long it keeps its card after a call before the idle residency comes back |
| `fallback`, `on` | a second card, and how it is launched there (`on: {"gpu0": {"api": "cuda", "command": […]}}`) |
| `spill` | also take its waiting calls on the fallback while its own card is busy with another tenant's job |

A **batch** tenant (`"kind": "batch"`) is one fresh process per job: `python script request.json out_dir`,
whose last stdout line is the job's JSON result; `settings` and `timeout_s` (default 7200 s).

### tenants and services

A **tenant** is what the scheduler places: one process (or a plugin tenant's parts) on its own card,
with an optional fallback card. A **service** is what callers name: a model, never a card. The
scheduler knows only tenants; when the config is loaded, each service becomes one or more server
tenants.

| | tenant | service |
|---|---|---|
| is | a server, a batch tenant or a plugin tenant, placed on a card | a name callers use, e.g. `qwen3-8b` |
| kinds | `server`, `batch`, or a plugin's | HTTP servers only |
| cards | its `device`, and optionally a `fallback` with its own launch there (`on`) | `backends`: one launch per card, each with its own `device`, `api`, `command`, `url`, … |
| called | `/engines/<tenant>/…`, first in its card's queue | `/v1/…` by the body's `model`, in creation order with other jobs |

`placement: "prefer"` (the default) makes one tenant named like the service: the first backend's card
is its own, the second backend's its fallback, launched as that backend says (so at most two
backends). `placement: "spread"` makes one tenant per backend, `<service>@<card>`, all running at once,
each call going to the least busy. `names`, and `slots`, `routes`, `linger_s`, `spill`, `poll` for
all of its backends, complete a service.

Use a service when callers name the model, or when it is launched differently on different cards (a
Vulkan build on one, a CUDA build on another); a tenant for a batch job, a plugin, or a server
called by name that runs the same way wherever it is.

`policy.idle.restore` lists what each card gets back when its queue is idle, on its own card only.

Any other `kind` is run by a plugin: see below.

## Calling it

The facade (`facade/app.py`) is the only network entry. Every route needs `Authorization: Bearer <token>`,
except from the addresses in `GPU_TRUSTED_CLIENTS` on `/engines` and `/v1`.

| route | |
|---|---|
| `GET /v1/models` | the model names, how many calls each takes at once, and each card's backend (framework, build, weights) |
| `POST /v1/…` | an OpenAI-style call, queued by its body's `model`; `"devices": ["gpu1"]` in the body limits it to those cards (and is removed before the server sees it). Streams back as the server writes. |
| `GET /v1/backends/<id>` | the checked facts of the process an `X-GPU-Backend` header names |
| `/engines/<tenant>/<path>` | a call to one tenant by name; these go first in their card's queue |
| `POST /rpc` `{"action", "arguments"}` | the job service itself (below) |
| `GET /files/jobs/…` | a job's output files |
| `GET /health` | the runtime's status |

RPC actions: `models`, `backend`, `engine_submit`, `batch_submit`, `job_get`, `jobs_get`, `job_wait`,
`job_cancel`, `resource_status`, `resource_acquire`, `resource_release`, `lend_submit`, `lend_get`,
`lend_heartbeat`, `lend_return`, `device_drain`. Every submission takes a `request_key`: the same key
with the same request returns the same job, with another request it is refused.

### Lending and draining

```bash
python3 runtime/gpu_borrow.py run --device gpu0 --hours 2 -- python train.py   # queued; only gpu0 visible
python3 runtime/gpu_borrow.py reserve --device gpu0 --hours 72 --label "fine-tune"   # exclusive; prints the env
python3 runtime/gpu_borrow.py release LEND_ID
python3 runtime/gpu_drain.py gpu1              # finish its queue, take nothing new, then empty it
python3 runtime/gpu_drain.py --resume gpu1
```

`--owner` (default user@host) says who borrows, `--label` what for; gpu_top and `gpu_borrow.py status`
show both beside the lend.

## Writing a tenant kind

A tenant whose `kind` is neither `server` nor `batch` is run by the module `<kind>_tenant` on the
Python path (or the one its `plugin` names), whose `Tenant` subclasses `gpu_runtime.TenantKind`:

```python
from gpu_runtime import TenantKind

class Tenant(TenantKind):
    jobs = ("render",)                         # the job kinds it runs

    @staticmethod
    def parts(spec):                           # its parts, each declared like a server's slot
        return {"encoder": spec["encoder"], "model": spec["model"]}

    def stages(self, kind, request):           # one device per stage, queued one after the other
        return [("encode", self.scheduler.lane_for(self.spec["encoder"])), ("sample", self.spec["model"]["device"])]

    def run(self, kind, stage, request, out, lane):
        ...                                    # start the part if need be (below), then use it
```

It starts its parts' processes with the scheduler's own steps (`_lease`, `_usable`, `_leave` and
`_started`, `_place`, `_using`, `_phase`, `_log`, `worker`) and hands each to it with
`scheduler.hold("<tenant>.<part>", device, worker)`; from then on the scheduler evicts, stops,
lends around and accounts for it like any other tenant. `first_slot`, `part_of`, `resident`, `home`,
`restore`, `status` and `ready` complete the contract (see `TenantKind`); `tests/pair_tenant.py` is
a small complete one. An application exposes its job kinds by subclassing `job_service.Service` and
adding the actions that submit them (`tests/app_fixture.py`), and runs `job_service.serve(Service, root)`.

## Environment

| variable | |
|---|---|
| `GPU_RUNTIME_LOCK`, `GPU_PHASE_LOCK` | the runtime's lease and its host-wide phase lock (default: in the state directory) |
| `HOST_DPKG_STATUS` | the host's dpkg status file, to compare its Mesa with the image's (default `/host/dpkg/status`) |
| `GPU_STATE`, `GPU_TOKEN_FILE`, `GPU_TRUSTED_CLIENTS` | the facade's state directory, token file and trusted addresses |
| `GPU_RUNTIME_SOCKET` | the socket the CLIs use (default `state/runtime.sock` beside the directory they run from) |

## Tests

```bash
python3 -m unittest discover -s tests
```

No GPU is needed: workers are fakes and the devices' readings are stubbed.

## License

Apache License 2.0, see `LICENSE`.
