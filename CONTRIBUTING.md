# Contributing

## Setup

```bash
git clone git@github.com:nlp-unibo/mamkit.git
cd mamkit
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

MAMKit supports Python 3.10 to 3.13.

Run local CI with `nox`:

| Command | Check |
|---|---|
| `nox` | Ruff and tests |
| `nox -s tests` | Tests with a 77% branch-coverage gate on every supported interpreter found locally |
| `nox -s docs` | Sphinx build |
| `nox -s package` | Wheel and sdist build, metadata check, and wheel import |

The sessions install CPU builds of `torch` and `torchaudio` before the package.
No check needs a GPU.

## Golden files

`tests/golden/` holds outputs recorded from the code before the Cinnamon refactoring.
Model files also hold the weights and inputs, so a rewritten model can load the same weights under its own parameter names.
`tests/golden/configs.json` describes every configuration as data.
Re-record a golden file only for an intended change of behavior, with `MAMKIT_RECORD=1 pytest <test>`, and state the change in the pull request.

The tests download small random-weight models from the Hugging Face Hub on their first run.

## Branch and pull-request flow

1. Branch from an updated `main` using `feat/`, `fix/`, `refactor/`, `test/`, `docs/`, or `chore/`.
2. Keep one logical change per branch, and add tests and public documentation together.
3. Run `nox`, plus any relevant optional session.
4. Open a pull request and complete the template.
5. Merge only when `All checks passed` succeeds and one review approves the change.
6. Prefer a squash or rebase merge, and delete the merged branch.

The recommended `main` ruleset requires pull requests, one approval, resolved conversations, linear history, and the status check `All checks passed`.
It also blocks force pushes and branch deletion.

## Documentation

Edit `docsrc/source/`.
`nox -s docs` writes the generated HTML to `docsrc/build/html`, and generated files stay uncommitted.
Every merge to `main` deploys that build through GitHub Pages.

## Release

1. Update `mamkit.__version__` in a pull request and merge it.
2. Tag the matching commit with `git tag -a vX.Y.Z -m "mamkit X.Y.Z"`.
3. Push the tag with `git push origin vX.Y.Z`.
4. `publish.yml` checks that the tag matches the version, builds the distributions, and publishes them through PyPI Trusted Publishing.

The repository owner must set GitHub Pages to use **GitHub Actions** as its source.
The owner must also configure PyPI Trusted Publishing for owner `nlp-unibo`, repository `mamkit`, workflow `publish.yml`, and environment `pypi`.
