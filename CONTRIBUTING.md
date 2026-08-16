# Contributing to ZimaBlue

Thanks for taking a look. ZimaBlue is early, which means the useful
contributions are large and structural as often as they are small.

## Setup

```bash
git clone https://github.com/JGalego/ZimaBlue
cd ZimaBlue
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
ruff format .
ruff check .
mypy
pytest
```

## What makes a good contribution here

**Extend through the registries, not through conditionals.** A new pool shape
is a new function with `@POOL_PRESETS.register("name")`. A new robot is a
composition of components. A new controller satisfies the `Controller`
protocol. If you find yourself adding a branch to an existing function to
support a new variant, the extension point is probably missing — say so in an
issue.

**Do not break determinism.** The contract is: same ZimaBlue version, same
platform, same scenario, same seed ⇒ bit-identical recording. That means:

- no `random`, `np.random.seed`, or unseeded generators — take a stream from
  the `RngTree` with a stable name
- no wall-clock reads in the stepping path
- no iteration over a set, or over a dict whose insertion order can vary
- no thread pools or parallel reductions inside a single run

`tests/test_determinism.py` will catch most violations. Please do not skip it.

**Prefer a small real model to a large fake one.** ZimaBlue would rather say
"flocculation is not modelled" in a docstring than ship a coefficient that
looks like physics and is not. If you simplify, name the simplification.

**Test observable behaviour.** Assert that the robot removes dirt, that a
seeded run reproduces, that a recording round-trips. Do not assert the internal
shape of a private array — that just makes refactoring expensive.

## Heavy dependencies

`pip install zimablue` must stay light: NumPy, Shapely, PyYAML, Typer, Rich.
Anything larger (matplotlib, and eventually Isaac Sim) goes behind an optional
extra and must be imported lazily, inside the function that needs it. Importing
`zimablue` must never import a GPU stack.

## Commit style

Conventional-ish prefixes (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), a
subject line under ~72 characters, and a body that says *why* when the
reason is not obvious from the diff.

## Regenerating the logo

The logo is generated from the `kidney` pool preset:

```bash
python tools/make_logo.py
```

If you change that preset's geometry, regenerate and commit both SVGs.

## Citing prior art

New models should say where they come from. Add the reference to
[`docs/references.md`](docs/references.md) with a resolving DOI, and verify it
against Crossref rather than trusting a citation you copied -- doing exactly
that to the first version of that file turned up a paper that does not exist, a
journal attributed to the wrong title, and an author list belonging to a
different group.

## Reporting a bug

For simulation bugs, include the scenario YAML (or the constructing code) and
the seed. That is usually enough to reproduce exactly — which is the entire
point of the determinism contract.
