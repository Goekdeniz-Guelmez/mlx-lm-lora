# Local paths and server lifecycle

This reference replaces the old tenant instructions; its filename is retained
for existing links. The PR #104 server is a local process with optional HTTP
bearer authentication. It has no tenant namespace or `tenant://` resolver.

`model` and `data` can be Hub IDs or server-local paths. A dataset directory
needs `train.jsonl` and may contain `valid.jsonl` and `test.jsonl`. Remote
clients cannot refer to their own filesystem as if it were the server's.

`judge` and `reference_model_path` in `extra_config` accept Hub IDs or local
model paths. `resume_adapter_file` and `reward_functions_file` must be existing
server-local files. The server resolves these paths and expands `~` before
launch because the trainer's working directory is its run directory.
Custom reward files execute Python in the trainer; use the user's intended file.

The default runs root is `~/.mlx-lm-lora/runs`; configure it with `--runs-dir`
or `MLX_LM_LORA_RUNS_DIR`. Each run has `config.yaml`, `run.json`, `train.log`,
an exit-code file when the wrapper exits normally, and an `adapters/` folder.
Use paths returned by the server. One active run is allowed per shared root;
different roots can launch independent runs on the same GPU.

Transports: stdio, or streamable HTTP at `/mcp` (`--transport http`). Default
HTTP binding is 127.0.0.1:8765. `MLX_LM_LORA_MCP_TOKEN` enables HTTP bearer
authentication; `--allowed-host` controls extra accepted Host headers.
`MLX_LM_LORA_PYTHON` selects the training Python; it must have MLX, MLX-LM and
the training dependencies installed. A separate server Python can host MCP.
For a source checkout, ensure the trainer can import that checkout through
its environment even after changing working directory.

Runs are detached and their state survives server restarts. Closing the server
does not stop training. Stop a requested run with `stop_training`, then shut
down the hosting server process separately if asked. Do not delete run artifacts
when stopping. Metal GPU access may require running outside a restricted sandbox.
