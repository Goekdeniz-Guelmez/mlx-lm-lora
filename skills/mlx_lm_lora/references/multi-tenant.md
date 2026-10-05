# Tenants, paths, and job state

Read this reference for tenant selection, local model/auxiliary paths, or
recovering jobs. Every training/status/log operation resolves its tenant at
the server boundary.

## Select a tenant

Use `configured_tenant_id` from capabilities for a pinned single-tenant
server. An authenticated shared server can derive the tenant from its token;
otherwise pass the user's explicit `tenant_id` outside the `config` object.
The server rejects mismatches with the pinned/authenticated tenant and enforces
its allow-list. Do not silently switch tenants to make a request succeed.

Credentials belong in server/client authentication settings, not in training
config values. This skill does not configure bearer tokens or sharing policy.

## Local references

A tenant workspace contains `inputs/`, `runs/`, and `artifacts/`. Use
`tenant://` paths for approved local models, reference/judge models, custom
reward files, resume weights, or artifact output:

```json
{
  "model": "org/model",
  "data": "org/dataset",
  "train": true,
  "train_mode": "sft",
  "resume_adapter_file": "tenant://inputs/adapter.safetensors",
  "adapter_path": "tenant://artifacts/resumed-run",
  "iters": 100
}
```

The path is relative to the selected tenant workspace. Relative `./` or `../`
inputs must also remain inside that workspace after resolution. Absolute
model/auxiliary inputs must lie under the tenant workspace or optional shared
input root. Outputs must stay inside the tenant workspace, never the shared
root. Path validation is separate from checking whether an input file exists.

The shared root allows local model/auxiliary inputs; it does not make local
`data` paths valid. Datasets must remain Hub repository IDs. Read
[datasets.md](datasets.md).

## Track the job

`mlx_lm_lora_start_training` returns a tenant-owned `job_id`, status, run
location, and artifact path. Reuse that tenant and job ID for status/log calls.
States are `queued`, `running`, `succeeded`, `failed`, and `cancelled`.

`mlx_lm_lora_list_training_runs` lists recent jobs (default 20, maximum 100).
`mlx_lm_lora_get_training_log` returns a bounded tail (default 12000 characters,
maximum 50000). A queued job may not have a log yet. Metadata stores the
normalized request and timestamps in the run directory.

Cancellation only applies before training starts. Resuming uses an explicit
adapter weights file and starts a new job; it is not restoration of the
previous job's optimizer or queue state. Report server-returned paths rather
than guessing where checkpoints or fused models were written.
