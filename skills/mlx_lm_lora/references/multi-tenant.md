# Tenants and local files

Use the tenant pinned by the MCP process for a local single-tenant agent. For a
shared server, use the user's explicit tenant only when it is authorized by
the server. Never move a job or artifact between tenants.

The training dataset is always a Hugging Face repository ID. Use `tenant://`
only for supported local tenant-owned auxiliary files:

```json
{
  "data": "org/dataset",
  "resume_adapter_file": "tenant://inputs/adapter.safetensors"
}
```

Hub model and dataset IDs such as `org/model` and `org/dataset` are passed
through unchanged. Do not put bearer tokens or credentials in the config.

The server returns a `job_id`; use that same ID with status and log tools while
polling. Artifacts and logs are tenant-scoped by the server.
