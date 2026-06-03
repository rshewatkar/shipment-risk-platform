# Contributing Guide

## Branch Strategy

main        ? stable, production-ready code only. Never push directly.
develop     ? integration branch for completed features
feature/*   ? one branch per feature (e.g. feature/data-generator)
fix/*       ? bug fixes (e.g. fix/missing-value-handling)
chore/*     ? maintenance tasks (e.g. chore/update-dependencies)

## Commit Message Format (Conventional Commits)

<type>: <short description>

Types:
  feat      ? new feature
  fix       ? bug fix
  chore     ? maintenance, dependencies, config
  docs      ? documentation only
  test      ? adding or fixing tests
  refactor  ? code change with no feature/fix
  ci        ? CI/CD pipeline changes

Examples:
  feat: add synthetic data generator for shipment records
  fix: handle missing carrier values in feature pipeline
  chore: pin numpy version to 1.26.4
  docs: add API endpoint documentation
  test: add unit tests for address risk scorer

## Pull Request Rules

1. Never merge directly to main without a PR
2. All CI checks must pass before merging
3. At least one self-review before merging

## Code Style

- Formatter  : black (line length 88)
- Import sort: isort (black profile)
- Linter     : flake8
- Type hints : required on all public functions
- Docstrings : required on all classes and public functions

Run before every commit:
  pre-commit run --all-files
