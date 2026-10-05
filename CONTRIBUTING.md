# Contributing

Thanks for contributing!

## Pull Request Requirements

Before opening a pull request, make sure that:

- All tests pass.
- `pre-commit` passes for every file changed, added, or updated by your PR.
- New functionality includes appropriate tests.
- Bug fixes include a regression test where practical.
- Documentation is updated when behavior, configuration, or public APIs change.
- The PR description clearly explains what changed and why.

## Local Checks

Run the test suite:

```bash
python -m unittest discover -s tests
```

Run pre-commit against the files changed in your branch:

```bash
pre-commit run --files $(git diff --name-only origin/main...HEAD)
```

To run pre-commit across the entire repository:

```bash
pre-commit run --all-files
```

## Testing Expectations

If your PR adds or changes behavior, add tests that cover:

- The intended behavior.
- Relevant edge cases.
- Failure or error paths when applicable.

Keep tests deterministic and avoid relying on network access, external services, or machine-specific state.

## Adding a Training Algorithm

Implement the algorithm's argument dataclass, training function, and evaluator in
`mlx_lm_lora/trainer/`. Register them together in `trainer/registry.py` using
`TrainingMode`: declare additional model/tokenizer inputs, CLI field aliases,
evaluation options, and returned result fields. The CLI uses this entry for mode
choices, model requirements, training, and evaluation; no dispatch branches in
`train.py` are needed. Omitted options retain the trainer or evaluator defaults,
so preserve intentional differences between algorithms.

Add algorithm-specific CLI defaults/options and dataset handling where needed.
Cover numerical behavior in trainer tests and option forwarding in
`tests/test_registry.py`, especially sampling, long-context, and memory settings.

Reuse `preference_utils.py` for DPO/CPO scoring and cached chunks,
`rollout_utils.py` for reward evaluation and bounded rollout scoring, and
`training_utils.save_adapters` for checkpoint saves. Keep each algorithm's loss,
normalization, and execution strategy explicit; preserve the evaluation boundary
after every backward chunk or scoring microbatch so activations can be released.

## Pull Request Checklist

- [ ] Tests pass locally.
- [ ] Pre-commit passes for all changed files.
- [ ] New or changed behavior is covered by tests.
- [ ] Documentation is updated where needed.
- [ ] The PR is focused and contains no unrelated changes.
