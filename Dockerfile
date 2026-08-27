# Difflet inference image for AWS Trainium (Trn2).
#
# Builds the same environment as scripts/setup_env.sh — Neuron runtime libs from
# the Neuron apt repo, Python stack pinned by requirements-neuron.lock — so the
# image shares compile-cache keys with hosts built by setup_env.sh.
#
# Build:
#   docker build -t difflet .
# Run (needs Neuron devices and a writable compile/weights cache):
#   docker run --rm \
#     --device /dev/neuron0 \
#     -v $HOME/.cache/difflet:/root/.cache/difflet \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     difflet run --model-id black-forest-labs/FLUX.1-dev --tp-degree 4 \
#       --height 1024 --width 1024 --prompt "a cat" --output /out/cat.png

FROM ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg2 git python3 python3-venv python3-pip \
 && . /etc/os-release \
 && curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
      | gpg --dearmor -o /usr/share/keyrings/neuron.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" \
      > /etc/apt/sources.list.d/neuron.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools \
 && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/aws/neuron/bin:${PATH}

WORKDIR /workspace/Difflet

# Install the pinned Python stack first so source edits don't bust this layer.
COPY requirements-neuron.lock ./
RUN python3 -m venv /workspace/venv \
 && /workspace/venv/bin/pip install --upgrade pip \
 && /workspace/venv/bin/pip install \
      --extra-index-url https://pip.repos.neuron.amazonaws.com \
      -r requirements-neuron.lock

COPY . .
# --no-deps: the lock already pins everything; re-resolving can downgrade
# Neuron wheels (see the dependency comment in pyproject.toml).
RUN /workspace/venv/bin/pip install -e ".[test]" --no-deps

ENV PATH=/workspace/venv/bin:${PATH}
ENTRYPOINT ["difflet"]
