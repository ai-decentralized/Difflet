# Setting up difflet.yottalabs.ai on a fresh trn3

This is the runbook for bringing the public Difflet server up on a **new** machine, from an empty
instance to `https://difflet.yottalabs.ai` answering. Follow the steps in order; each one ends with
a check. What the gateway does and every fleet field are in [gateway.md](gateway.md).

```text
browser ──HTTPS──▶ Caddy :443 (TLS, blocks two routes) ──▶ difflet gateway 127.0.0.1:8000
                                                             │  routes by "model"
                         ┌──────────┬──────────┬─────────────┼──────────┬──────────┐
                     FLUX :8100  Qwen :8101  Wan :8102  Hunyuan :8103  LTX-2 :8104  H3 :8105
                     device 0    device 1    device 2    device 3      device 4     device 5
```

The page is the gateway's own `/` (same origin, no CORS, no static host). Everything the fleet
needs (weights, compiled artifacts, logs) lives on the instance-store RAID at `/mnt/nvme`.

> **Instance store is wiped when the instance is stopped** (a reboot keeps it). Never stop the
> instance before the event; back up `/mnt/nvme/hub` and the compile caches ([step 10](#10-back-up-before-you-stop-the-instance))
> so the next machine restores in minutes instead of re-downloading and recompiling.

Paths used throughout:

| Path | What |
|---|---|
| `~/Difflet` | The repo checkout, installed editable into the device venv |
| `/opt/aws_neuronx_venv_torch_neuronx_2_14_0` | The device venv (DLAMI TorchNeuron, native neuron backend) |
| `/mnt/nvme/hub` | The Hugging Face hub cache (all six models' weights) |
| `/mnt/nvme/difflet-fleet/fleet.yaml` | The fleet file |
| `/mnt/nvme/difflet-fleet/cache/<slug>` | Each backend's compiled artifacts |
| `/mnt/nvme/difflet-fleet/runs/<slug>/serve.out` | Each backend's console log |
| `/mnt/nvme/difflet-fleet/fleetctl` | Wrapper every step below uses to run `difflet gateway` |

## 0. Before you start

- A **trn3.48xlarge** on the Neuron DLAMI (Ubuntu 24.04, Python 3.12) with 4 instance-store NVMe
  disks. The fleet uses devices 0–5 (4 cores each); nothing else may hold them.
- An **Elastic IP** associated with the instance (a plain public IP changes on stop/start, and DNS
  points at it).
- A **security group** with 22 (from your IP), **80 and 443 from anywhere** inbound. Do not open
  8000 or 8100–8105.
- Access to the `yottalabs.ai` zone in **Cloudflare** (to point `difflet` at the Elastic IP).
- A **Hugging Face token** whose account has accepted the FLUX.1-dev license (a gated repo).
- GitHub access to `ai-decentralized/Difflet`.

## 1. Mount the NVMe disks as one RAID0 at /mnt/nvme

```bash
lsblk -d -o NAME,SIZE,MODEL                    # the four 1.7T "Amazon EC2 NVMe Instance Storage"
DISKS=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/ {print "/dev/"$1}')
echo $DISKS                                    # must list exactly four disks, never the EBS root
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 --run $DISKS
sudo mkfs.ext4 -F -E nodiscard -L difflet-nvme /dev/md0
sudo mkdir -p /mnt/nvme
sudo mount -o noatime /dev/md0 /mnt/nvme
sudo chown ubuntu:ubuntu /mnt/nvme
```

Survive a reboot (the array and the mount come back; `nofail` keeps the boot from hanging after a
stop/start, when the disks are blank again and this step must be redone):

```bash
sudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf
sudo update-initramfs -u
echo 'LABEL=difflet-nvme /mnt/nvme ext4 defaults,noatime,nofail 0 2' | sudo tee -a /etc/fstab
```

Check: `df -h /mnt/nvme` shows about 6.9T.

Use the disk names `lsblk` prints: the EBS root is usually `nvme0n1` but the order is not
guaranteed. `mdadm --create` asks for confirmation if a disk still has an old signature; answer `y`
only after checking that `$DISKS` is the four instance-store disks.

## 2. Code and venv

The fleet runs the **native** neuron backend (`DIFFLET_BACKEND=neuron`, which the launcher sets).
It needs the DLAMI's TorchNeuron venv, `/opt/aws_neuronx_venv_torch_neuronx_2_14_0` (torch 2.14,
torch-neuronx 2.14), plus difflet and its dependencies.

> Not `scripts/setup_env.sh`'s `.venv`: that one is torch-xla, and every backend dies at startup
> with `AssertionError: Unknown backend type neuron` (no `neuron` process-group backend).

```bash
cd ~ && git clone git@github.com:ai-decentralized/Difflet.git && cd Difflet   # main has the gateway
V=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin
# difflet's dependencies the SDK venv lacks (diffusers, transformers, accelerate, uvicorn, ...),
# pinned by the native lock; the dry run must not list torch or any neuron package
DEPS=$($V/python - <<'EOF'
import tomllib, re, importlib.metadata as md
for d in tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]:
    try:
        md.version(re.split(r"[<>=!~\[; ]", d, maxsplit=1)[0])
    except md.PackageNotFoundError:
        print(d.split(";")[0].strip())
EOF
)
echo "$DEPS"
if [ -n "$DEPS" ]; then   # empty on a rerun: everything is installed
  $V/pip install --dry-run -c requirements-neuron-native.lock $DEPS | grep "^Would install"
  $V/pip install -c requirements-neuron-native.lock $DEPS
fi
# in the lock but not in pyproject: FLUX's and Wan's T5 tokenizers need sentencepiece
$V/pip install -c requirements-neuron-native.lock sentencepiece pyzstd ninja
$V/pip install -e . --no-deps
(cd /tmp && $V/python -c "import torch_neuronx, diffusers, sentencepiece, difflet.serving.gateway.app; print('ok')")
```

The last check runs from `/tmp` on purpose: from `~/Difflet`, `import difflet` succeeds even when
difflet is not installed, and the backends (which run in `/mnt/nvme/difflet-fleet/runs/<slug>`)
then fail with `No module named 'difflet'`. `pip check` afterwards still lists `datasets` and
`pipenv` conflicts; difflet uses neither.

Create the wrapper the rest of this guide uses (it activates the venv and points every backend at
the shared hub cache, so manual runs, cron and systemd behave the same; the SDK venv has no
`difflet` script, hence `python -m`):

```bash
mkdir -p /mnt/nvme/difflet-fleet /mnt/nvme/hub
cat > /mnt/nvme/difflet-fleet/fleetctl <<'EOF'
#!/usr/bin/env bash
# fleetctl up|down|status|serve [args] -- difflet gateway on this host's fleet file
set -euo pipefail
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
export HF_HUB_CACHE=/mnt/nvme/hub
action=$1; shift
exec python -m difflet.cli.main gateway "$action" --config /mnt/nvme/difflet-fleet/fleet.yaml "$@"
EOF
chmod +x /mnt/nvme/difflet-fleet/fleetctl
```

## 3. Weights

Either restore them (fast) or download them.

**Restore from a backup** ([step 10](#10-back-up-before-you-stop-the-instance)):

```bash
aws s3 sync s3://<bucket>/difflet-fleet/hub /mnt/nvme/hub
```

**Or download** (`difflet download` fetches only the files each model needs, e.g. 144 GB of
MiniMax-H3's 498 GB repo):

```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
export HF_HUB_CACHE=/mnt/nvme/hub
huggingface-cli login                          # the token with the FLUX.1-dev license
for m in black-forest-labs/FLUX.1-dev Qwen/Qwen-Image Wan-AI/Wan2.1-T2V-14B-Diffusers \
         hunyuanvideo-community/HunyuanVideo Lightricks/LTX-2 MiniMaxAI/MiniMax-H3; do
  python -m difflet.cli.main download --model-id "$m" || echo "FAILED: $m"
done
```

All six are 419 GB; in us-east-2 they downloaded in about 4 minutes. Check: every model printed `weights ready`, and
`du -sh /mnt/nvme/hub/models--*` lists six models. The backends run with `HF_HUB_OFFLINE=1`, so a
model that is missing here fails at startup instead of downloading.

## 4. Fleet file

Start from the example and make two edits: every backend reads the hub cache `fleetctl` sets (the
example's per-model `HF_HUB_CACHE` paths are from the old machine), and the page is same-origin
(no `cors_origins`).

```bash
cd ~/Difflet
sed -e '/^ *env: {HF_HUB_CACHE:/d' -e '/cors_origins:/d' \
    examples/serving/fleet.yaml > /mnt/nvme/difflet-fleet/fleet.yaml
grep -n "HF_HUB_CACHE\|cors_origins" /mnt/nvme/difflet-fleet/fleet.yaml   # prints nothing
/mnt/nvme/difflet-fleet/fleetctl status
```

`status` validates the file (an invalid one is refused, with the field named) and lists the six
models as `down`. Keep `gateway.host: 127.0.0.1`: only Caddy may reach the gateway.

## 5. Compiled artifacts and backends

If you have a backup of the compile caches, restore it first; with the same
device venv (the same torch-neuronx and neuronx-cc versions) the artifacts are reused as they are:

```bash
aws s3 sync s3://<bucket>/difflet-fleet/cache /mnt/nvme/difflet-fleet/cache
```

Start the backends (in `tmux`):

```bash
/mnt/nvme/difflet-fleet/fleetctl up
```

`up` starts the six backends one at a time. Each one loads its weights, **compiles if its
`cache_dir` is empty**, runs a smoke generation, and only then opens its port. Follow along with:

```bash
tail -f /mnt/nvme/difflet-fleet/runs/<slug>/serve.out     # slug: flux, qwen, wan, ...
```

- Without restored caches, the first `up` compiles every model and takes hours. `up` waits up to
  an hour per backend; when that runs out it stops with an error, but **the backend keeps
  running**. Run `fleetctl up` again: it skips running backends and continues with the next one.
- A backend that exits is reported with the last 40 lines of its log; fix the cause and rerun
  `up`.
- `up` refuses a device another process holds. `neuron-ls` shows who holds what.

Check: `fleetctl status` shows all six `ready`, and
`curl -s 127.0.0.1:8000/ready` (after step 6) answers `"status": "ready"`.

## 6. The gateway as a service

```bash
sudo tee /etc/systemd/system/difflet-gateway.service >/dev/null <<'EOF'
[Unit]
Description=Difflet gateway (difflet.yottalabs.ai)
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/mnt/nvme

[Service]
User=ubuntu
WorkingDirectory=/mnt/nvme/difflet-fleet
ExecStart=/mnt/nvme/difflet-fleet/fleetctl serve
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now difflet-gateway
```

Check: `curl -s 127.0.0.1:8000/v1/models | jq '.data[] | {id, difflet_status}'`.

Restarting the gateway (`sudo systemctl restart difflet-gateway`, for a fleet-file or code change)
reloads no model: the backends are separate processes. It does reset the per-client limits, which
live in the gateway's memory.

Do **not** run `fleetctl up` from a systemd unit: the backends would live in that unit's cgroup and
die with it. Start them by hand (step 5) or from cron (step 8).

## 7. Caddy (TLS) and DNS

Install Caddy from its apt repository:

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
```

`/etc/caddy/Caddyfile`:

```caddyfile
difflet.yottalabs.ai {
	# The page never lists or deletes video jobs. Open, these let anyone read every user's
	# prompts (GET /v1/videos?model=...) and cancel anyone's job (DELETE /v1/videos/{id}).
	@job_list {
		method GET
		path /v1/videos
	}
	@job_delete {
		method DELETE
		path /v1/videos/*
	}
	respond @job_list 403
	respond @job_delete 403

	reverse_proxy 127.0.0.1:8000 {
		# behind Cloudflare (proxied): the gateway counts the real client, not the edge node;
		# only safe while 80/443 accept Cloudflare's IP ranges alone (else the header can be forged)
		header_up X-Forwarded-For {http.request.header.CF-Connecting-IP}
		# pass streamed video bytes on at once; no read timeout may be added: a request can
		# wait in its model's queue and run for many minutes
		flush_interval -1
	}
}
```

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

**DNS, in Cloudflare (proxied, orange cloud):** an `A` record `difflet` → the Elastic IP, Proxied.
Cloudflare then hides the origin and absorbs floods, and Caddy still gets a Let's Encrypt
certificate itself: the HTTP-01 challenge reaches it through Cloudflare on port 80. Three
settings make the pieces fit:

- **SSL/TLS mode *Full (strict)*** in the zone. *Flexible* makes Cloudflare call the origin over
  HTTP, Caddy redirects that to HTTPS, and the browser loops on the redirect.
- **The security group admits 80 and 443 from [Cloudflare's IP ranges](https://www.cloudflare.com/ips/) only**,
  not `0.0.0.0/0`. The `header_up` line trusts `CF-Connecting-IP`; a client reaching the origin
  directly could forge it. (`0.0.0.0/0` works too, at the cost that a direct client can pick the
  IP the per-IP limit counts.) Never open the ports the backends' ranks listen on: each backend's
  torch.distributed store binds `0.0.0.0` on a random high port, without authentication.
- **Cloudflare ends a proxied request at 100 s** (`524`). The page stays within it: images take
  seconds and wait at most 30 s in their queue, and videos are async jobs the page polls. API
  calls to `/v1/videos/sync` for a video longer than that fail.

With the record **DNS only (grey cloud)** instead, remove the `header_up` line: without Cloudflare
the header is absent, the gateway would see every client as `127.0.0.1`, and the per-IP limit
would be shared by everyone.

If the site answers `522`, Cloudflare cannot reach the origin: the security group does not admit
80/443 yet (on this host, `curl -s http://169.254.169.254/latest/meta-data/security-groups` with an
IMDSv2 token lists the groups actually attached). `525` right after opening the ports means Caddy
has no certificate yet because its earlier attempts failed and it is backing off;
`sudo systemctl reload caddy` makes it try again at once.

Check, from any machine:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://difflet.yottalabs.ai/   # 200 (HEAD gets 405: / is GET only)
curl -s https://difflet.yottalabs.ai/ready | jq .status                # "ready"
curl -s -o /dev/null -w '%{http_code}\n' \
  'https://difflet.yottalabs.ai/v1/videos?model=Lightricks/LTX-2'     # 403
```

## 8. Restart backends that die

A `difflet serve` process that dies stays down (its worker restarts itself, the process does not).
Re-run `up` every 5 minutes; it skips running backends, and `flock` keeps two runs from overlapping:

```bash
( crontab -l 2>/dev/null; echo '*/5 * * * * flock -n /tmp/difflet-fleet-up.lock /mnt/nvme/difflet-fleet/fleetctl up >> /mnt/nvme/difflet-fleet/watchdog.log 2>&1' ) | crontab -
```

`up` stops at the first backend that fails to start, so the ones after it in the fleet file wait
until it is fixed; `watchdog.log` shows which one.

Restart a single backend (its process group; `fleetctl down` stops all six):

```bash
kill -TERM -- -$(cat /mnt/nvme/difflet-fleet/runs/<slug>/serve.pid)
/mnt/nvme/difflet-fleet/fleetctl up
```

## 9. End-to-end check

1. Open `https://difflet.yottalabs.ai/info`: six models, all ready, each with its estimated time
   and an example request; `https://difflet.yottalabs.ai` is the generation form.
2. Generate one image (FLUX) and one video (HunyuanVideo, the fastest video model) from the page.
3. The same from a terminal:

   ```bash
   curl -sS -X POST https://difflet.yottalabs.ai/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "black-forest-labs/FLUX.1-dev",
          "messages": [{"role": "user", "content": "a small red sailboat on a calm blue lake"}],
          "extra_body": {"height": 1024, "width": 1024, "seed": 42}}' \
     | jq -r '.choices[0].message.content[0].image_url.url' | cut -d, -f2- | base64 -d > image.png
   ```

4. Optional, after a code change: re-measure each model's `estimate` (time two step counts through
   the gateway, solve `base_s + per_step_s * steps`) and update the fleet file; the page's times
   and the wait thresholds use it.

## 10. Back up before you stop the instance

Stopping the instance erases `/mnt/nvme`. Save what is slow to rebuild:

```bash
aws s3 sync /mnt/nvme/hub                 s3://<bucket>/difflet-fleet/hub
aws s3 sync /mnt/nvme/difflet-fleet/cache s3://<bucket>/difflet-fleet/cache
cp /mnt/nvme/difflet-fleet/fleet.yaml ~/fleet.yaml.bak              # EBS root survives a stop
```

On the next machine, steps 3 and 5 restore from these instead of downloading and compiling.

## Known gaps for a public event

- The API accepts up to 200 steps (the page caps at 50); a 200-step Wan request holds its device
  for about half an hour, and the wait estimates assume the default steps.
- No prompt filtering: anyone can type anything, and the page carries the yottalabs.ai name.
- One backend per model (the fleet file refuses a repeated model), so devices 6–15 stay idle;
  under load, image requests get `429` once their model's queue (8) is full or a request has
  waited 30 s.
- The per-client and per-IP limits live in the gateway's memory and reset when it restarts.
