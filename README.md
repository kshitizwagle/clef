# clef

A web UI and API for [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash), Cloudflare's 9B
decision model, running locally on [llama.cpp](https://github.com/ggml-org/llama.cpp). Tested on an
AMD Radeon RX 9070 XT under WSL2 with ROCm.

A decision model does not write text. It takes a state and typed questions and returns a
probability for every allowed answer. This project serves it through llama.cpp's `/v1/systemone`
endpoint (the Jev/SystemOne API) and adds a UI with two tabs:

- **Ask**: enter a state and one or more questions, each with a list of candidate answers. Each
  question is sent to the model as a `choice` question. Results show the predicted answer and every
  answer's probability, with questions ranked by confidence.
- **Evaluate CSV**: paste labelled examples, or load them from a file. Each row is sent to the
  model and its prediction is compared with the row's correct answer. Shows accuracy overall, by
  confidence band and by question, and every row ranked by confidence, with a results CSV to
  download.

## Tested on

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT, 16 GB (RDNA4, `gfx1201`) |
| CPU | AMD Ryzen 7 7700X, 8 cores / 16 threads |
| Memory available to WSL | 7.7 GB |
| Windows | Windows 11 Pro, build 26300 |
| GPU driver (Windows) | AMD Software: Adrenalin Edition, driver version 32.0.31041.1004 (dated 2026-08-17) |
| WSL | 2.7.14.0, kernel 6.18.33.2-microsoft-standard-WSL2 |
| Linux | Ubuntu 24.04.5 LTS |
| ROCm | 10.0.0 (`amdrocm10.0-gfx1201` 10.0.0-4), HIP 7.15, HSA runtime 1.21 |
| WSL GPU bridge | librocdxg 1.2.2 (`rocdxg-roct_1.2.2_amd64.deb`) |
| Compiler | AMD clang 23.0.0 (ROCm), GCC 13.3.0, CMake 3.28.3 |
| llama.cpp | `main` at [`a55e952b8`](https://github.com/ggml-org/llama.cpp/commit/a55e952b8) (build 11376), HIP backend |
| Model | [`ggml-org/Clef-Flash-GGUF`](https://huggingface.co/ggml-org/Clef-Flash-GGUF) `Q8_0`, 8.98 GiB |

With this setup, all 33/33 layers are offloaded to the GPU (8,169 MiB of model weights in VRAM,
about 5 GB left free), and the server is ready about 20 seconds after start once the model is
downloaded.

## Setup: AMD Radeon GPU on WSL2

These are the steps used on the machine above. Under WSL2 the GPU is reached through `/dev/dxg`
(DirectX), not the Linux `amdgpu` driver, so ROCm needs the librocdxg bridge, and the Linux
`amdgpu` DKMS driver must **not** be installed. Use HIP/ROCm rather than Vulkan: inside WSL,
Vulkan only gets the slower Dozen (Vulkan-on-D3D12) layer.

### 1. Windows

1. Install the AMD Software: Adrenalin Edition driver for your GPU from
   [AMD Drivers and Support](https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-9000-series/amd-radeon-rx-9070-xt.html).
   AMD lists the driver versions supported for WSL in the
   [WSL support matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html).
2. Update WSL from PowerShell: `wsl --update`.
3. In Ubuntu, check that the GPU device exists:

   ```
   ls -l /dev/dxg
   ```

   If it is missing, WSL needs updating.

### 2. ROCm 10.0

Add AMD's ROCm repository and install the runtime for your GPU target, following
[Install AMD ROCm 10.0.0](https://rocm.docs.amd.com/en/latest/install/rocm.html):

```
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | \
  gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null

sudo tee /etc/apt/sources.list.d/amdrocm-stable.sources << EOF
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
EOF

sudo apt update
sudo apt install amdrocm10.0-gfx1201
```

`gfx1201` is the RX 9070 XT. For another GPU, use its target (`rocminfo` shows it once the GPU is
visible, and AMD's support matrices list them).

### 3. librocdxg (the WSL GPU bridge)

ROCm on WSL loads [librocdxg](https://github.com/ROCm/librocdxg) to reach the GPU through
`/dev/dxg`. Install the prebuilt package from its
[releases](https://github.com/ROCm/librocdxg/releases):

```
wget https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
sudo apt install ./rocdxg-roct_1.2.2_amd64.deb
```

It installs `/opt/rocm/lib/librocdxg.so.1.2.2`. See
[librocdxg issue #74](https://github.com/ROCm/librocdxg/issues/74) for another report of the
RX 9070 XT working this way.

Check that ROCm sees the GPU:

```
/opt/rocm/bin/rocminfo | grep -E 'gfx1201|Marketing Name'
```

Expected:

```
  Name:                    gfx1201
  Marketing Name:          AMD Radeon RX 9070 XT
```

`rocm-smi` does not work under WSL, so don't use it to check.

### 4. ROCm development files

The runtime package alone is not enough to compile against HIP. Building llama.cpp needs the HIP
and hipBLAS headers and CMake configs, which are in the development metapackage:

```
sudo apt install amdrocm-core-dev10.0-gfx1201
```

Without it, CMake fails with `does not contain the HIP runtime CMake package`.

### 5. Build llama.cpp with HIP

```
sudo apt install cmake ninja-build build-essential libssl-dev

git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp
cd ~/src/llama.cpp
HIPCXX="$(/opt/rocm/bin/hipconfig -l)/clang" HIP_PATH="$(/opt/rocm/bin/hipconfig -R)" \
  cmake -S . -B build -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

`libssl-dev` lets llama-server download models from Hugging Face over HTTPS (`-hf`). Clef support
is in llama.cpp `main` (it was added in
[PR #29831](https://github.com/ggml-org/llama.cpp/pull/29831)).

`HSA_OVERRIDE_GFX_VERSION` was not needed.

### 6. Run this app

Requires [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/kshitizwagle/clef && cd clef
uv run clef
```

Open http://127.0.0.1:8000. The first start downloads the model (about 10 GB) into
`~/.cache/huggingface`.

To confirm the GPU is used, look for these lines in `~/clef-server.log`:

```
using device ROCm0 (AMD Radeon RX 9070 XT) (0000:03:00.0) - 13753 MiB free
load_tensors: offloaded 33/33 layers to GPU
load_tensors:        ROCm0 model buffer size =  8168.77 MiB
```

## How it runs

On startup the app runs llama-server
(`~/src/llama.cpp/build/bin/llama-server -hf ggml-org/Clef-Flash-GGUF:Q8_0 -ngl 99 -lv 4`) as a
child process and waits until the model is loaded before accepting requests. llama-server's output
is shown in the terminal and written to `~/clef-server.log`. Stopping the app (Ctrl+C) stops
llama-server; if the app is killed, llama-server receives SIGTERM and exits too.

If a model server is already answering at `CLEF_UPSTREAM`, the app uses it instead of starting one.

| Env var | Default | Purpose |
|---|---|---|
| `CLEF_UPSTREAM` | `http://127.0.0.1:8080` | llama-server address (host and port it is started on) |
| `CLEF_MANAGE_SERVER` | `1` | `0` to never start llama-server, only connect to `CLEF_UPSTREAM` |
| `CLEF_LLAMA_SERVER` | `~/src/llama.cpp/build/bin/llama-server` | llama-server binary |
| `CLEF_MODEL` | `ggml-org/Clef-Flash-GGUF:Q8_0` | Model passed to `-hf` |
| `CLEF_LLAMA_ARGS` | | Extra llama-server arguments, e.g. `"-b 4096 -ub 4096"` |
| `CLEF_SERVER_LOG` | `~/clef-server.log` | llama-server log file |
| `CLEF_START_ATTEMPTS` | `3` | Starts to try when llama-server crashes during startup |
| `CLEF_READY_TIMEOUT` | `1800` | Seconds to wait for the model to load (first run downloads ~10 GB) |
| `CLEF_EVAL_CONCURRENCY` | `4` | Rows evaluated at once (llama-server runs 4 slots by default) |
| `CLEF_UPSTREAM_TIMEOUT` | `120` | Seconds to wait for a prediction |
| `CLEF_HOST` / `CLEF_PORT` | `127.0.0.1` / `8000` | Where the web app listens |

The app has no authentication. Keep `CLEF_HOST` on `127.0.0.1` unless the network is trusted.

## Input size limit

By default a request can be at most **512 tokens**, counting the state, the question and the
options. Clef evaluates the whole prompt in one physical batch, and llama-server's default batch
size (`--ubatch-size`) is 512. Larger requests fail with:

```
input (1007 tokens) is too large to process. increase the physical batch size (current batch size: 512)
```

The 262,144-token `n_ctx` in the log is the base model's training context and does not apply
here. To accept longer inputs, raise the batch size, at the cost of more GPU memory:

```
CLEF_LLAMA_ARGS="-b 4096 -ub 4096" uv run clef
```

How far the batch size can go on a 16 GB card has not been measured yet; the compute buffer is
138 MiB at the default 512.

For comparison, Cloudflare's reference code defaults to 16,384 tokens (`max_length`), Clef on
Workers AI has a 64,000-token context, and Jev accepts 64k tokens for the state plus all
questions.

## Known issues

- **Startup crash under WSL.** The ROCm runtime sometimes segfaults while creating the GPU context
  and prints `[CreateContext] fail 11`. It happened in 2 of about 12 starts on the machine above. When llama-server crashes like this during startup, the app starts it again, up to
  `CLEF_START_ATTEMPTS` times. If llama-server exits while the app is running, the app logs the
  error and shuts down rather than serving without a model.
- **No image input.** Clef-Flash reads images and video, but llama.cpp does not support image input
  for Clef yet.

## CSV format

| Column | Contents |
|---|---|
| `context` | The situation the model reasons about |
| `question` | The question to answer |
| `answer` | The correct answer |
| `options` | Optional. Candidate answers separated by any punctuation, e.g. `TRUE;FALSE;NOT GIVEN` or `yes\|no`. If empty or missing, the options are the distinct answers to the same question in the file |

The file is read with pandas. Columns can be separated by any punctuation or a tab: common
separators (`,` tab `;` `|`) are tried first, and the first one that yields the `context`,
`question` and `answer` headers is used. Rows copied from a spreadsheet are tab-separated, so they
can be pasted directly. A field containing the separator must be in double quotes, with any quotes
inside it doubled (`""`). Extra fields at the end of a row, such as trailing separators, are
ignored.

In the `options` cell, the parser picks the separator whose split includes the row's `answer`, so
a comma inside an answer like `Yes, definitely` is not mistaken for a separator. Header names are
case-insensitive, and answers are matched to options case-insensitively. Invalid rows (an empty
field, an answer that is not one of the options, fewer than 2 options) are all reported before
anything is sent to the model. Row numbers match a spreadsheet: the header is row 1. See
[`src/clef/static/sample.csv`](src/clef/static/sample.csv).

## API

`POST /api/predict`

```json
{
  "state": "Checkout has been failing for every customer for the last hour.",
  "questions": [
    {"question": "Which team should handle this?", "answers": ["billing", "technical", "shipping"]}
  ]
}
```

Returns `results`: one entry per question, sorted by `confidence` (highest first). Each entry has
`question`, `prediction`, `confidence`, and `answers` (each `answer` with its `probability`,
highest first).

`POST /api/evaluate` takes a CSV as the request body (`Content-Type: text/csv`) and streams
NDJSON: a `start` line with the row count, a `row` line per row as it finishes, then a `summary`
line. An invalid CSV returns `400` with every problem listed.

`GET /api/health` reports whether the model server is reachable.

## Test

```
uv run pytest
```

## License

This project is released under the [MIT License](LICENSE).

It does not include or redistribute model weights or llama.cpp; both are downloaded or built
separately and have their own licenses:

- [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) model weights: Apache-2.0
  (post-trained from [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), also Apache-2.0)
- [llama.cpp](https://github.com/ggml-org/llama.cpp): MIT
- [ROCm](https://github.com/ROCm/ROCm) and [librocdxg](https://github.com/ROCm/librocdxg): see
  AMD's repositories
