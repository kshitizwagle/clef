# clef

A web UI and FastAPI service for Clef-Flash, served by llama.cpp's `/v1/systemone` endpoint.

Enter a state and one or more questions, each with a list of candidate answers. Each question is
sent to the model as a `choice` question. Results show the predicted answer and every answer's
probability, with questions ranked by confidence.

## Run

```
uv run clef
```

Open http://127.0.0.1:8000. On startup the app runs llama-server
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
| `CLEF_LLAMA_ARGS` | | Extra llama-server arguments, e.g. `"-c 8192"` |
| `CLEF_SERVER_LOG` | `~/clef-server.log` | llama-server log file |
| `CLEF_READY_TIMEOUT` | `1800` | Seconds to wait for the model to load (first run downloads ~10 GB) |
| `CLEF_UPSTREAM_TIMEOUT` | `120` | Seconds to wait for a prediction |
| `CLEF_HOST` / `CLEF_PORT` | `127.0.0.1` / `8000` | Where the web app listens |

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

`GET /api/health` reports whether the model server is reachable.

## Test

```
uv run pytest
```
