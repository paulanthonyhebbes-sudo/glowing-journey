# BitNet on the Homelab — Runlist

Running Microsoft's **BitNet b1.58 2B-4T** (1.58-bit ternary LLM) via **bitnet.cpp** in an LXC container on Proxmox, served on the LAN with `llama-server`.

This runlist reflects the setup as it actually worked, including the three fixes the official model needed (Phase 6).

## Quick start — server command

Inside the container (`pct enter 110` from the Proxmox host):

```bash
cd ~/BitNet && conda activate bitnet-cpp
./build/bin/llama-server -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --host 0.0.0.0 --port 8080 -t 2 \
  --jinja --chat-template-file bitnet-template.jinja \
  --override-kv tokenizer.ggml.eos_token_id=int:128009,tokenizer.ggml.pre=str:llama-bpe
```

- Web chat: `http://192.168.1.236:8080`
- OpenAI-compatible API: `http://192.168.1.236:8080/v1`
- Requires the relu² patch (Phase 6.1) and the template file (Phase 6.2).

> ⚠️ `--override-kv` must be given **once**, with values comma-separated. If repeated, only the last one is used.

## Target hardware

| Component | Spec |
|---|---|
| Host | HP 260 G3 DM mini PC, Proxmox VE |
| CPU | Intel i5-7200U — 2 cores / 4 threads, 15 W, **AVX2 ✅** |
| RAM | 16 GB DDR4 (runs at 2133 MHz — the CPU's maximum) |
| GPU | None (CPU-only inference) |

## Container plan

| Setting | Value | Why |
|---|---|---|
| Type | Unprivileged LXC, Ubuntu 24.04 | Sees host AVX2 directly, no VM overhead |
| CT ID | 110 | Change if taken |
| Cores | 4 | Allows testing `-t 2` vs `-t 4` |
| CPU limit | 3 | Always leaves headroom on the host |
| CPU units | 50 | Half default priority — Pi-hole/Tailscale win under contention |
| RAM / swap | 4096 MB / 1024 MB | Model needs < 2 GB; compile needs headroom |
| Disk | 20 GB on `local-lvm` | PyTorch dependencies are bulky |

> **Login:** the container is created without a root password. Use `pct enter 110` from the host. To use the web console instead, set one with `pct exec 110 -- passwd`.

---

## Phase 1 — Create the container (Proxmox host)

- [ ] Update and find the Ubuntu 24.04 template

```bash
pveam update
pveam available --section system | grep ubuntu-24.04
```

- [ ] Download it (use the exact filename shown above)

```bash
pveam download local ubuntu-24.04-standard_24.04-2_amd64.tar.zst
```

- [ ] Check free IDs and storage names

```bash
pct list
pvesm status
```

- [ ] Create the container

```bash
pct create 110 local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst \
  --hostname bitnet \
  --cores 4 --cpulimit 3 --cpuunits 50 \
  --memory 4096 --swap 1024 \
  --rootfs local-lvm:20 \
  --net0 name=eth0,bridge=vmbr0,ip=dhcp \
  --unprivileged 1 --features nesting=1 \
  --onboot 0
```

- [ ] Start and enter it

```bash
pct start 110
pct enter 110
```

**✅ Checkpoint:** prompt shows `root@bitnet`

---

## Phase 2 — Build tools (inside container)

- [ ] Update and install

```bash
apt update && apt upgrade -y
apt install -y git cmake clang build-essential wget curl jq
```

- [ ] Verify

```bash
clang --version | head -1
cmake --version | head -1
grep -o avx2 /proc/cpuinfo | head -1
```

**✅ Checkpoint:** clang 18.x, cmake 3.2x, and `avx2` printed

---

## Phase 3 — Python environment (Miniforge, Python 3.10)

- [ ] Install Miniforge

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
exec bash
```

- [ ] Create and activate the environment

```bash
conda create -n bitnet-cpp python=3.10 -y
conda activate bitnet-cpp
```

**✅ Checkpoint:** prompt starts with `(bitnet-cpp)`

> Every time you re-enter the container, run `cd ~/BitNet && conda activate bitnet-cpp` before any BitNet command. Check the prompt shows `(bitnet-cpp) root@bitnet:~/BitNet#`.

---

## Phase 4 — Clone BitNet and download the model

- [ ] Clone and install dependencies (CPU-only PyTorch)

```bash
cd ~
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

- [ ] Download the model (~1.2 GB)

```bash
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
```

> If `huggingface-cli` isn't found, use `hf download` with the same arguments.

**✅ Checkpoint:** `ls models/BitNet-b1.58-2B-4T` shows `ggml-model-i2_s.gguf`

---

## Phase 5 — Build bitnet.cpp

- [ ] Run setup (slow on 2 cores — tens of minutes)

```bash
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

> Progress: from the host, `pct exec 110 -- tail -f /root/BitNet/logs/compile.log`. If it fails: `tail -30 logs/compile.log`.

- [ ] Snapshot the clean build (**Proxmox host**)

```bash
pct snapshot 110 clean-build
```

---

## Phase 6 — Fix the model (required)

Out of the box, the official GGUF rambles, repeats and never stops. Three fixes are needed. See **Troubleshooting** for how each was found.

### 6.1 Patch the activation function (relu²)

bitnet.cpp hardcodes SiLU for this model; it should use squared ReLU ([microsoft/BitNet#602](https://github.com/microsoft/BitNet/issues/602)). Tested against llama.cpp submodule commit `390c30775`.

- [ ] Back up and locate the line

```bash
cp 3rdparty/llama.cpp/src/models/bitnet.cpp 3rdparty/llama.cpp/src/models/bitnet.cpp.bak
grep -n "LLM_FFN_SILU" 3rdparty/llama.cpp/src/models/bitnet.cpp
```

Expect **one** result (around line 133). If there's more than one, stop and check.

- [ ] Patch it (replace `133` with the line number shown)

```bash
sed -i '133s/LLM_FFN_SILU/model.arch == LLM_ARCH_BITNET_B158 \&\& model.type == LLM_TYPE_2B ? LLM_FFN_RELU_SQR : LLM_FFN_SILU/' 3rdparty/llama.cpp/src/models/bitnet.cpp
grep -n "LLM_FFN_RELU_SQR" 3rdparty/llama.cpp/src/models/bitnet.cpp
```

- [ ] Rebuild (only the changed parts recompile)

```bash
cmake --build build --config Release -j 2
```

### 6.2 Create the chat template

The GGUF's built-in chat format doesn't match what the model expects.

- [ ] Paste this block in one go

```bash
cat > ~/BitNet/bitnet-template.jinja << 'EOF'
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{- 'System: ' + message['content'] + '<|eot_id|>' -}}
{%- elif message['role'] == 'user' -%}
{{- 'User: ' + message['content'] + '<|eot_id|>' -}}
{%- elif message['role'] == 'assistant' -%}
{{- 'Assistant: ' + message['content'] + '<|eot_id|>' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- 'Assistant: ' -}}
{%- endif -%}
EOF
```

### 6.3 Tokenizer overrides

Applied at server start (see Quick start): sets `<|eot_id|>` (ID `128009`) as the stop token and the Llama 3 pre-tokenizer (`llama-bpe`). The second removes the `GENERATION QUALITY WILL BE DEGRADED` warning.

- [ ] Snapshot the working build (**Proxmox host**)

```bash
pct snapshot 110 working-relu2
```

---

## Phase 7 — Start and verify the server

- [ ] Start the server with the **Quick start** command (top of this file)

- [ ] Second shell (`pct enter 110`) — chat test

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":100}'
```

**✅ Pass:** `"finish_reason":"stop"`, `prompt_tokens` ≈ 14, short answer

- [ ] Token-level check — model should emit `128009` and stop

```bash
curl -s http://localhost:8080/completion -H "Content-Type: application/json" \
  -d '{"prompt":"User: What is the capital of France?<|eot_id|>Assistant: ","n_predict":20,"n_probs":1,"temperature":0}' \
  | jq -c '.completion_probabilities[] | [.id, .token]'
```

**✅ Pass:** ends with `[13,"."]` then `[128009,""]`

- [ ] Access from other devices
  - [x] Browser on main PC: `http://192.168.1.236:8080`
  - [ ] Phone via Tailscale (subnet router): same address
  - [ ] AnythingLLM → Generic OpenAI provider, base URL `http://192.168.1.236:8080/v1`

---

## Phase 8 — Benchmark

- [ ] 2 threads

```bash
python utils/e2e_benchmark.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -n 128 -p 512 -t 2
```

- [ ] 4 threads

```bash
python utils/e2e_benchmark.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -n 128 -p 512 -t 4
```

### Results

`pp512` = prompt processing speed · `tg128` = text generation speed

| Machine | Threads | pp512 (tok/s) | tg128 (tok/s) | Notes |
|---|---|---|---|---|
| HP 260 G3 (i5-7200U) | 2 | 13.78 ± 0.02 | 14.99 ± 0.09 | **Best — use for server.** Pre-patch (SiLU) |
| HP 260 G3 (i5-7200U) | 4 | 11.25 ± 0.47 | 11.96 ± 0.14 | Pre-patch. Run with `cpulimit 3` (throttled) |
| HP 260 G3 (i5-7200U) | 2 | | | Post-patch (relu²) rerun |
| HP 260 G3 (i5-7200U) | 4 | | | Optional rerun with `cpulimit 4` |
| Main PC (Ryzen 7 5800X3D) | | | | For comparison |

**Run details:** I2_S model, 1.10 GiB, 2.41B params, CPU backend, llama-bench build `390c30775`, tested 11 Oct 2026.

**Speeds seen in normal use (2 threads):**

| Tool | Prompt (tok/s) | Generation (tok/s) |
|---|---|---|
| `run_inference.py` (llama-cli) | 49.9–51.9 | 13.7–15.0 |
| `llama-server` (14-token prompt) | ~43 | ~14.8 |

**Observations:**

- Generation at ~15 tok/s on 2 threads is roughly 2–3x human reading speed, on a 15 W 2017 laptop chip with no GPU.
- 2 threads beat 4: the i5-7200U has only 2 physical cores (hyperthreads share execution units), and the 4-thread run was also capped by the container's `cpulimit 3`.
- The benchmark uses `n_batch = 1`, feeding the prompt one token at a time, so pp512 ≈ tg128. With batching, prompt processing ran at ~50 tok/s.
- The `Q1_0` label is llama-bench's naming for the format; the model is the official I2_S GGUF.
- Benchmarks above were run before the relu² patch. The activation change should not materially affect speed, but this hasn't been re-measured yet.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Console rejects root login | Container created without a password | `pct enter 110` from host, or `pct exec 110 -- passwd` |
| `can't open file '/root/utils/...'` | Wrong folder / env not active | `cd ~/BitNet && conda activate bitnet-cpp` |
| `curl: (7) Failed to connect` | Server not running | Start it in another shell first |
| `GENERATION QUALITY WILL BE DEGRADED` warning | GGUF lacks `tokenizer.ggml.pre` | `--override-kv tokenizer.ggml.pre=str:llama-bpe` |
| Reply unrelated to prompt; `prompt_tokens: 3` | `--chat-template bitnet` treated as a literal template in this build | Use `--jinja --chat-template-file bitnet-template.jinja` |
| `DEPRECATED: argument '--override-kv' specified multiple times` | Only the last override is applied | One `--override-kv`, comma-separated values |
| Correct first sentence, then endless repetition; `finish_reason: "length"`; token check shows no `128009` | Runtime uses SiLU instead of relu² ([#602](https://github.com/microsoft/BitNet/issues/602)) | Apply the Phase 6.1 patch and rebuild |
| `perl: warning: Setting locale failed` during apt | Container has no `en_US.UTF-8` locale | Harmless. Optional: `apt install -y locales && locale-gen en_US.UTF-8` |

---

## Next steps

- [ ] Run `llama-server` as a systemd service so it starts automatically
- [ ] DHCP reservation for `192.168.1.236` (address is currently dynamic)
- [ ] Rerun the benchmark post-patch
- [ ] Repeat the benchmark on the main PC for comparison
- [ ] Comment on [microsoft/BitNet#602](https://github.com/microsoft/BitNet/issues/602) confirming the fix on Linux / i5-7200U
- [ ] Write up results for LinkedIn / GitHub
- [ ] Use energy and efficiency findings in Unit 41 (Digital Sustainability)

## Useful commands

| Task | Command (Proxmox host) |
|---|---|
| Enter container | `pct enter 110` |
| Stop / start | `pct stop 110` / `pct start 110` |
| Roll back to working build | `pct rollback 110 working-relu2` |
| Roll back to clean (unpatched) build | `pct rollback 110 clean-build` |
| Check resource use | `pct status 110 --verbose` |

## References

- [microsoft/BitNet on GitHub](https://github.com/microsoft/BitNet)
- [BitNet-b1.58-2B-4T GGUF on Hugging Face](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf)
- [microsoft/BitNet#602 — FFN uses SiLU instead of relu²](https://github.com/microsoft/BitNet/issues/602)
