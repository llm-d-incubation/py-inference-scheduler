# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Image for the Mooncake shared KV tier (Ray Serve workers AND the store master).
#
# ray-llm 2.56.0 bakes vllm 0.22.0, which ships the multi-file mooncake/store
# connector (introduced in vLLM 0.21.0), and its ray.llm imports match that
# vllm's layout — pairing verified locally (full-stack imports + connector test
# suite green). So the only addition is the Mooncake wheel, pinned so the
# master and every replica client run the same version.
#
# The SAME image runs the mooncake master (configs/mooncake/master.yaml just
# overrides the command with mooncake_master) — one artifact, zero
# master/client version skew.
#
# Build + push (from the repo root):
#   docker build -f configs/mooncake/worker.Dockerfile \
#     -t <your-registry>/ray-llm-mooncake:2.56.0 .
#   docker push <your-registry>/ray-llm-mooncake:2.56.0

FROM rayproject/ray-llm:2.56.0-py312-cu130

# mooncake_master and the RDMA transfer engine link against libnuma at runtime.
RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends libnuma1 \
    && sudo rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --no-cache-dir "mooncake-transfer-engine==0.3.11.post1"

# The mooncake wheel links the CUDA 12 runtime; this cu130 image only ships
# CUDA 13's. Install cu12's and register it with the loader (libcuda.so.1
# itself is injected by GKE at run time, so it can't be checked at build).
RUN uv pip install --system --no-cache-dir nvidia-cuda-runtime-cu12 \
    && echo /home/ray/anaconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \
       | sudo tee /etc/ld.so.conf.d/nvidia-cu12-runtime.conf \
    && sudo ldconfig

# Sanity: connector imports, cu12 runtime resolves, master binary starts.
RUN python -c "import vllm, ray, ctypes; \
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector; \
    ctypes.CDLL('libcudart.so.12'); \
    print('image OK: ray', ray.__version__, '/ vllm', vllm.__version__)" \
    && (mooncake_master --help > /dev/null 2>&1 || [ $? -ne 127 ])
