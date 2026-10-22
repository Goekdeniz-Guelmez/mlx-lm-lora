# DSLA: Directional and Similarity-aware Latent Alignment

## Data and settings

Use prompt/chosen/rejected rows retaining nonempty prompt and response spans after tokenization. `dsla_loss` selects dpo (default), cpo or orpo. Only DSLA-DPO loads a frozen reference. The model needs a compatible residual-stream backbone; logits are not hidden states.

## MCP example

Pass these flat arguments to `start_training`; use the same arguments with
`check_training_config` first. Hub IDs below are illustrative.

```json
{
  "model": "org/policy-model",
  "data": "org/training-dataset",
  "train_mode": "dsla",
  "train_type": "lora",
  "iters": 100,
  "batch_size": 1,
  "num_layers": 1,
  "max_seq_length": 512,
  "fuse": false,
  "extra_config": "{\"dsla_loss\": \"dpo\", \"latent_weight\": 0.1, \"latent_margin\": 0.05, \"latent_gamma\": 10.0, \"latent_variant\": \"both\", \"latent_pooling\": \"answer_mean\", \"latent_layer\": \"final\"}"
}
```

## Best practices

Establish the corresponding plain preference baseline, then isolate the effect of latent alignment. Track latent_loss, similarity/direction losses and margins alongside preference metrics. Choose latent_variant from similarity/direction/both, pooling from answer_mean/last_token/last_k_mean/prompt_answer_mean, and layer from final/middle/late or a valid index. Keep latent_weight/margin nonnegative and latent_gamma positive. Compare pooling/layer choices on held-out data rather than assuming a universal best layer.

## Memory efficiency

Hidden-state retention adds activation pressure to pair scoring. Use short but complete examples, checkpointing, small minibatches and accumulation; recurrence_chunk_size helps recurrent models. `efficient_long_context` is explicitly rejected because pooling needs complete-sequence representations. QAT is wired. Switching underlying objective changes semantics and must not be done silently to remove a reference copy.

See the [shared memory guide](memory-efficiency.md) for tuning order and the
[MCP contract](config.md) for defaults, monitoring and iteration-budget limits.
