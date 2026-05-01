# Contributing

Thanks for your interest in ClustGrade-RKO. This project is the public
companion code for an academic article; we welcome bug reports, fixes, and
small feature contributions.

## Development workflow

1. Fork and clone the repository.
2. Create a feature branch off `main`: `git checkout -b feat/your-change`.
3. Install dev dependencies: `poetry install --with dev`.
4. Make your changes. Add or update tests as appropriate.
5. Run the full local check before pushing:

   ```bash
   poetry run ruff check .
   poetry run black --check .
   poetry run mypy src rko
   poetry run pytest -q
   ```

6. Open a pull request against `main`. CI must pass and one approving review
   is required before merge.

## Branch protection on `main`

`main` is protected. The maintainer-side configuration on GitHub is:

- Pull request required before merge.
- One approving review required.
- All CI status checks must pass.
- Force pushes and branch deletions are disabled.
- Linear history (rebase or squash merge only).

## Code style

- Python ≥ 3.12.
- Line length 100 (Black) / 120 (Ruff).
- Type hints on all public functions and non-trivial helpers.
- Docstrings follow the Google style; flag O() complexity on hot paths.

## Reporting bugs

Open an issue with: a minimal reproducer, the dataset (or a small synthetic
substitute), the expected and observed behavior, and your environment
(`python --version`, `poetry show numpy scipy scikit-learn`).
