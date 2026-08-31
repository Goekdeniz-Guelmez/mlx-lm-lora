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

## Pull Request Checklist

- [ ] Tests pass locally.
- [ ] Pre-commit passes for all changed files.
- [ ] New or changed behavior is covered by tests.
- [ ] Documentation is updated where needed.
- [ ] The PR is focused and contains no unrelated changes.
