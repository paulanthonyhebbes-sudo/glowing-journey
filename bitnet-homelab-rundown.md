# Running a 1.58-bit LLM on a 2017 Mini PC

**Project:** Microsoft BitNet b1.58 2B-4T on a Proxmox homelab
**Date:** 11 October 2026
**Status:** Working — served on the LAN, answering correctly and stopping cleanly

## Summary

I set up Microsoft's BitNet b1.58 2B-4T, a 2.4-billion-parameter language model whose weights are only -1, 0 or +1, on an HP 260 G3 mini PC with an Intel i5-7200U (2 cores, 15 W, no GPU). It runs in a Proxmox LXC container and is served across my network through an OpenAI-compatible API.

It generates text at about **15 tokens per second** on two CPU threads.

Out of the box, the official model file rambled, repeated itself and never stopped. I traced this through three separate problems: a chat template mismatch, a command-line option that was silently ignored, and a confirmed upstream bug where bitnet.cpp uses the wrong activation function for this model. After patching the source and rebuilding, the model answers correctly and stops on its own.

## What BitNet is

Most language models store each weight as a 16-bit number. Quantised models, like the Q4 files used in LM Studio or Ollama, compress those weights to around 4 bits after training. BitNet models are trained from the start with each weight restricted to -1, 0 or +1. That's about 1.58 bits per weight (log₂3).

This means most of the multiplication in the model becomes addition, subtraction or skipping, which CPUs handle well. Microsoft reports speedups of 2.37x–6.17x and energy reductions of 71.9%–82.2% on x86 CPUs compared with full-precision models. The model file is 1.10 GiB.

bitnet.cpp is Microsoft's inference engine for these models, built on llama.cpp with custom kernels.

## What I built

### Hardware

| Component | Spec |
|---|---|
| Machine | HP 260 G3 DM mini PC |
| CPU | Intel Core i5-7200U, 2 cores / 4 threads, 15 W, AVX2 |
| RAM | 16 GB DDR4 at 2133 MHz |
| GPU | None |
| Hypervisor | Proxmox VE (also running Pi-hole and a Tailscale subnet router) |

### Container

An unprivileged Ubuntu 24.04 LXC container, chosen over a VM because it sees the host CPU's AVX2 instructions directly and has no guest-kernel overhead.

| Setting | Value |
|---|---|
| Cores / CPU limit | 4 / 3 |
| CPU units | 50 (half priority, so DNS stays responsive under load) |
| RAM / disk | 4 GB / 20 GB |

### Software stack

- clang 18 and CMake from Ubuntu's repositories
- Miniforge with a Python 3.10 environment
- bitnet.cpp (llama.cpp submodule at commit `390c30775`) with the I2_S kernel
- `llama-server` exposing a web chat page and an OpenAI-compatible API on port 8080

## Benchmark results

Measured with bitnet.cpp's `e2e_benchmark.py` (llama-bench), 512-token prompt and 128 generated tokens. These runs were done before the activation patch described below.

| Threads | Prompt processing (tok/s) | Text generation (tok/s) |
|---|---|---|
| **2** | **13.78 ± 0.02** | **14.99 ± 0.09** |
| 4 | 11.25 ± 0.47 | 11.96 ± 0.14 |

In normal use, where prompts are processed in batches rather than one token at a time, prompt processing ran at around 43–52 tokens per second and generation at 13.7–15.0.

### What the numbers show

- **15 tokens per second is faster than reading speed**, on a 2017 laptop chip with no GPU.
- **Two threads beat four.** The i5-7200U has only two physical cores; the extra hyperthreads compete for the same hardware. The four-thread run was also capped by the container's CPU limit of 3, so it isn't a perfectly fair comparison.
- **The benchmark understates prompt speed.** It feeds the prompt one token at a time (`n_batch = 1`), which is why prompt and generation speeds came out almost the same.

## Debugging the output

This took most of the session and taught me the most.

### Problem 1: replies unrelated to the question

**Symptom:** asked the bat-and-ball question, the model produced looping text about counting words in a sentence. The official `run_inference.py` chat script gave similar results.

**Test:** a plain prompt, "The capital of France is", gave "Paris", then repeated itself. So the model was computing something meaningful and the weights and kernels were at least partly working.

**First fix attempt:** restarting the server with `--chat-template bitnet`. The API response showed `prompt_tokens: 3` for a question that should be around 14 tokens. In this build, the option appears to have been treated as a literal template, so the model only received the word "bitnet".

**Fix:** a Jinja template file matching BitNet's training format (`User: …<|eot_id|>Assistant: `), loaded with `--jinja --chat-template-file`. After this, `prompt_tokens` was 14 and the first sentence of each answer was correct.

### Problem 2: an override that was silently ignored

**Symptom:** the model still never stopped. Text like "Paris.The capital…", with no space after the full stop, suggested a hidden stop token was being produced but not recognised.

**Fix attempt:** two `--override-kv` options, one setting `<|eot_id|>` (ID 128009, confirmed with the `/tokenize` endpoint) as the stop token and one setting the Llama 3 pre-tokenizer.

**What went wrong:** the server log showed `DEPRECATED: argument '--override-kv' specified multiple times… (only last value will be used)`. Only the pre-tokenizer setting had been applied. Combining both into one comma-separated option fixed that, and removed the "GENERATION QUALITY WILL BE DEGRADED" warning, but the model still didn't stop.

### Problem 3: the wrong activation function

**Test:** I asked the server for the ID of every generated token (`n_probs` via the `/completion` endpoint, filtered with `jq`). After "Paris." the next token was `791` ("The"). There was no hidden stop token. My theory was wrong: the model wasn't trying to stop at all.

**Cause:** a search turned up [microsoft/BitNet issue #602](https://github.com/microsoft/BitNet/issues/602). The official 2B-4T model is designed to use squared ReLU (relu²) as its activation function, but bitnet.cpp hardcodes SiLU. The model runs, but with subtly wrong maths in every layer, producing plausible but degraded text. The reporter's environment used the same llama.cpp submodule commit as mine, `390c30775`.

**Fix:** a one-line patch to `3rdparty/llama.cpp/src/models/bitnet.cpp`, applying relu² for the b1.58 2B model only, followed by an incremental rebuild with `cmake --build`.

**Result:** the token check now ends `[13,"."]`, `[128009,""]`. The model produces its end-of-turn token straight after "Paris." and stops.

### Before and after

| | Before fixes | After fixes |
|---|---|---|
| "What is the capital of France?" | Correct first sentence, then repeated until the 100-token cap | "The capital of France is Paris." then stops |
| Bat-and-ball question | Looping, off-topic text | Correct equation, correct working, correct answer ($0.05) |
| `finish_reason` | `length` | `stop` |

The bat-and-ball question is a well-known trick question where the intuitive answer ($0.10) is wrong. The patched model set up `x + (x + 1.00) = 1.10`, solved it step by step and got $0.05.

## What I learned

- **How ternary models differ from quantised ones.** BitNet is trained with three-valued weights, not compressed afterwards, so you can't convert an existing model into BitNet.
- **CPU features matter for virtualisation.** AVX2 has to be visible to the guest. LXC containers see the host CPU directly; Proxmox VMs need the CPU type set to `host` or `x86-64-v3`.
- **Hyperthreading doesn't help much for LLM inference** on a two-core chip.
- **Chat templates are part of the model.** A model wrapped in the wrong format can look broken even when it isn't.
- **Read the logs.** The ignored override was only visible in a deprecation warning at startup.
- **Test one change at a time and inspect the raw data.** Token-level output disproved my hidden-token theory and pointed to the real cause.
- **"Working" isn't the same as "correct".** The model loaded, ran fast and produced English throughout. Only its behaviour showed something was wrong underneath.

## Still to do

- Run the server as a systemd service so it starts automatically
- Reserve the container's IP address in DHCP
- Rerun the benchmark after the patch
- Benchmark the same model on my main PC (Ryzen 7 5800X3D) for comparison
- Test access from my phone over Tailscale and from AnythingLLM
- Comment on issue #602 confirming the fix on Linux with an i5-7200U

## References

- [microsoft/BitNet](https://github.com/microsoft/BitNet)
- [BitNet-b1.58-2B-4T GGUF on Hugging Face](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf)
- [microsoft/BitNet#602 — FFN uses SiLU instead of relu²](https://github.com/microsoft/BitNet/issues/602)
