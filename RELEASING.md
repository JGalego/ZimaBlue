# Releasing

Publishing runs through GitHub Actions with **PyPI Trusted Publishing**, so
there are no API tokens stored in this repository. GitHub mints a short-lived
OIDC token for the job and PyPI verifies it against a publisher you configure
once on each index.

## One-time setup

### 1. Create the environments in GitHub

Settings → Environments → New environment, twice: `testpypi` and `pypi`.

On `pypi`, add yourself under **Required reviewers**. That turns a real release
into a deliberate click rather than something a stray tag can do. Leave
`testpypi` unprotected so dry runs stay cheap.

### 2. Register the publisher on TestPyPI

Go to <https://test.pypi.org/manage/account/publishing/> and add a **pending**
publisher — "pending" is what lets you claim a name that does not exist yet:

| Field | Value |
|---|---|
| PyPI project name | `zimablue` |
| Owner | `JGalego` |
| Repository name | `ZimaBlue` |
| Workflow name | `release.yml` |
| Environment name | `testpypi` |

### 3. Register the publisher on PyPI

Same form at <https://pypi.org/manage/account/publishing/>, with the
environment name set to `pypi`.

Both entries must match the workflow exactly. A mismatch fails at upload with
an `invalid-publisher` error, which is the system working.

### 4. Add the Codecov token

Public repositories can upload tokenlessly, but it is rate-limited and fails
intermittently. Get the upload token from
<https://app.codecov.io/gh/JGalego/ZimaBlue/settings> and add it as a repository
secret named `CODECOV_TOKEN` (Settings → Secrets and variables → Actions).

Without it the coverage badge simply stays stale; nothing else breaks.

## Dry run to TestPyPI

Actions → Release → **Run workflow** → target `testpypi`.

That builds the distributions, checks the metadata with `twine --strict`,
installs the wheel into a clean virtualenv and runs the CLI, uploads to
TestPyPI, then installs *from* TestPyPI in a fresh job and runs the CLI again.

Do this at least once before a real release. It catches the failures that never
show up in development, because a git clone has the `scenarios/` directory and
the dev extras that a wheel may not.

## Real release

```bash
# 1. Bump the version. One file -- pyproject reads it from here.
$EDITOR src/zimablue/_version.py

# 2. Move the CHANGELOG's Unreleased entries under the new heading.
$EDITOR CHANGELOG.md

# 3. Land it.
git commit -am "chore: release v0.2.0"
git push

# 4. Tag. The workflow checks the tag against _version.py and stops if they
#    disagree, so this cannot publish 0.1.0 under a v0.2.0 tag.
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0
```

The tag runs the whole chain: build → TestPyPI → install from TestPyPI → PyPI.
The final step waits on the `pypi` environment, so it sits there until you
approve it.

## Notes

- **A version can never be reused.** PyPI refuses to accept a second upload of
  the same version even after a delete, so a broken release is fixed by
  releasing again with a higher number, not by cleaning up.
- **TestPyPI has its own accounts** and is periodically pruned. It is a
  rehearsal space, not a mirror.
- **`skip-existing: true` on the TestPyPI job** means re-running a workflow
  will not fail on files already uploaded.
- Installing from TestPyPI needs `--extra-index-url https://pypi.org/simple/`,
  because NumPy and Shapely live on the real index.
