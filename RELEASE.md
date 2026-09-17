# SOMA-X Release Checklist

This checklist covers public `py-soma-x` package releases from the public-safe
GitHub mirror.

## One-time PyPI setup

Configure Trusted Publishing for both PyPI and TestPyPI before creating a
release tag.

Use these publisher settings for the PyPI project `py-soma-x`:

- Owner: `NVlabs`
- Repository: `SOMA-X`
- Workflow: `pypi.yml`
- Environment: `pypi`

Use the same settings on TestPyPI with environment `testpypi`.

Protect the `pypi` and `testpypi` GitHub environments so publishing requires
maintainer approval. Do not configure long-lived PyPI API tokens for this
workflow.

## One-time Hugging Face setup

The v0.2.3 and later releases use Hugging Face Trusted Publishing. A maintainer
with Write access to `nvidia/SOMA-X` must add these claims under the repository's
**Settings -> Trusted Publishers** page:

- Provider: GitHub Actions
- Repository: `NVlabs/SOMA-X`
- Workflow: `pypi.yml`
- Branch: unset, because releases run from immutable version tags

Protect the `huggingface` GitHub environment so the first production sync
requires maintainer approval. The workflow requests a short-lived,
repository-scoped token through OIDC; do not configure a long-lived `HF_TOKEN`.

After initial setup and after every change to Hugging Face authentication or
publication code, run `pypi.yml` manually with `hf_git_smoke` enabled. The
smoke job uses the configured Trusted Publisher to create, verify, and remove
an ephemeral `ci-oidc-*` tag on the existing Hub snapshot. It does not upload
assets, publish packages, or require a package-version bump.

## Release steps

1. Merge the public-release prep MRs into internal `main`.
2. Cut or refresh the minor-line internal release branch, for example
   `release-0.3`. Patch releases reuse the same minor-line branch and create a
   new patch tag from that branch.
3. Confirm `setup.cfg` and `soma/__init__.py` both contain the intended package
   version, for example `0.3.0`. Run the internal versioned-data preflight on
   this release checkout; the internal Linux CI job also runs it automatically.
   Review any warning and update the cumulative history before preparing the
   public diff. Unchanged data does not need another snapshot. The authoring
   tools and their detailed procedure are maintained only in the internal tree.
4. Post the exact public file diff, exact Hugging Face publish set, and proposed
   public-facing GitHub commit message to the release issue.
5. Obtain explicit human approval for all three review items before creating
   the public GitHub commit. Any subsequent change to those inputs requires a
   new review.
6. Mirror the public-safe release branch to public GitHub using the approved
   commit subject and body.
7. Confirm the generated public mirror candidate passed the internal
   public-release validation gate before pushing.
8. Confirm the public GitHub Actions build job passes on the release branch.
9. Run the `pypi.yml` `hf_git_smoke` workflow-dispatch preflight and confirm
   that the ephemeral Hub tag is created, verified, and removed. This is
   mandatory after an automation change and before the first release that uses
   the changed automation.
10. Create the release tag from the public-safe release branch:

   ```bash
   git tag -a vX.Y.Z -m "SOMA-X X.Y.Z"
   git push origin vX.Y.Z
   ```

11. Approve the protected `huggingface` environment, then verify the workflow
   publishes and immutably tags the exact manifest on `nvidia/SOMA-X`.
   The job repeats the ephemeral-tag preflight before uploading the release
   snapshot, so a Git authentication failure cannot occur after a large upload.
12. Verify the workflow downloads the Hub tag and passes file-set and SHA-256
   checks before the TestPyPI job becomes eligible.
13. Verify the tag-triggered workflow publishes to TestPyPI.
14. Approve the protected `pypi` environment only after TestPyPI verification.
15. Verify PyPI shows the new `py-soma-x` release.
16. Verify the GitHub Pages changelog under `/stable/` and `/vX.Y/` shows the
    new release section. For patch releases, make sure the `release-X.Y` docs
    workflow completed; if the tag workflow was metadata-only or the branch
    workflow was cancelled, manually rerun the docs workflow on `release-X.Y`.
17. Record release links for the GitHub tag, Hub tag/manifest, PyPI release,
    docs, and validation artifact.

## Why the PyPI publish waits for the Hugging Face tag

The workflow runs its jobs in one chain: Hub publish, Hub verification,
TestPyPI, PyPI, GitHub Release. This coupling is deliberate. The package
defaults to Hub revision `vX.Y.Z` in `soma/assets.py` and downloads that tag on
first use, and PyPI releases are immutable. A wheel published before the Hub tag
exists would fail for every user at first import, so PyPI must not publish until
the Hub tag exists and matches the release manifest.

The risk is that a Hugging Face outage blocks the PyPI release even when PyPI
itself is healthy. On 2026-09-02 the Hub's OIDC token exchange failed for hours
(`invalid_grant: fetch failed` / `This operation was aborted` from
`hf auth token`) while the Hub, GitHub, and PyPI were all up. The fallback
below is how v0.3.0 shipped that day.

Only the Hub *tag state* is required, not the Hub *publish job*. Dispatching
`pypi.yml` with `tag` set and `hf_revision` empty skips the Hub publish and
recovery jobs; the verification job then downloads the existing Hub tag
anonymously and, if the manifest matches, the TestPyPI, PyPI, and GitHub
Release jobs run on PyPI Trusted Publishing, which is independent of Hugging
Face.

## Dispatch ref for `workflow_dispatch` runs

The `testpypi` and `pypi` environments only accept `v*.*.*` tag refs, while the
`huggingface` environment also accepts `main`. Therefore:

- `hf_git_smoke` preflight: dispatch from public `main` or from a tag.
- Any dispatch that should reach TestPyPI or PyPI (recovery, outage fallback):
  dispatch at the release tag ref, for example `--ref vX.Y.Z`. A dispatch from
  `main` fails at the TestPyPI job with "Branch main is not allowed to deploy".

## Interrupted Hugging Face release recovery

If the exact release snapshot reached Hub `main` but its immutable tag was not
created, do not bump the package version. Fix and smoke-test the automation,
then dispatch `pypi.yml` at the release tag ref with:

- `tag`: the existing public GitHub release tag, such as `v0.2.3`.
- `hf_revision`: the exact already-published Hub commit SHA.
- `hf_git_smoke`: disabled.

The recovery job uses the same short-lived OIDC credential to preflight Git
authentication, verify the Hub manifest at `hf_revision`, create the missing
release tag, and resume Hub verification, TestPyPI, PyPI, and the GitHub
Release. No personal Hugging Face token is required.

## Hugging Face OIDC outage fallback

Use this only when the Hub's OIDC exchange is down but the Hub itself accepts
uploads, and only with explicit maintainer approval recorded on the release
issue, because it deviates from the token policy above and skips step 9.

1. Confirm the failure is Hub-side, not a configuration change: from any host,
   `POST https://huggingface.co/oauth/token` (token-exchange grant) with an
   unsigned JWT whose `iss` is `https://token.actions.githubusercontent.com`
   returns `This operation was aborted` after about 10 s during the outage,
   while other issuers get an instant "No trusted publisher configured" reply.
   A claims or publisher misconfiguration produces an instant, specific error
   instead; fix that rather than using this fallback.
2. Create and push the release tag (step 10). When the tag run reaches the
   `huggingface` environment gate, reject the deployment so the OIDC publish
   job never runs. The run ends as failed with nothing uploaded.
3. On a maintainer machine, log in to Hugging Face with a fine-grained token
   scoped to `nvidia/SOMA-X` (`repo.write`), check out the public release tag
   with LFS content, and run the same tools the workflow uses:

   ```bash
   python -m pip install "huggingface_hub==<pin from pypi.yml>"
   HF_STAGE_DIR="$(mktemp -d)"
   python tools/ci/package_hf_assets.py --output "$HF_STAGE_DIR" \
     --release-tag vX.Y.Z --public-commit <40-character-public-commit>
   python tools/ci/verify_hf_assets.py "$HF_STAGE_DIR"
   python tools/ci/publish_hf_assets.py --stage "$HF_STAGE_DIR" \
     --repo-id nvidia/SOMA-X --release-tag vX.Y.Z
   ```

   The publish tool refuses to overwrite an existing tag with a different
   manifest. The personal token never enters GitHub; revoke or let it expire
   afterwards.
4. Verify independently without credentials:

   ```bash
   HF_DOWNLOAD_DIR="$(mktemp -d)"
   HF_TOKEN= hf download nvidia/SOMA-X --revision vX.Y.Z \
     --local-dir "$HF_DOWNLOAD_DIR"
   python tools/ci/verify_hf_assets.py --allow-download-cache "$HF_DOWNLOAD_DIR"
   ```

5. Dispatch `pypi.yml` at the release tag ref with `tag: vX.Y.Z`,
   `hf_revision` empty, `hf_git_smoke` disabled. The workflow re-verifies the
   Hub tag, publishes to TestPyPI, waits for `pypi` approval, publishes to
   PyPI, and creates the GitHub Release. Continue with steps 13 to 17.
6. Record on the release issue that the Hub upload used a personal token and
   that step 9 was skipped. Before the next release, re-run the `hf_git_smoke`
   preflight so the OIDC automation is proven again.

## Local checks

Run these from the release branch before tagging:

```bash
python tools/ci/check_release_version.py --expected vX.Y.Z
HF_STAGE_DIR="$(mktemp -d)"
python tools/ci/package_hf_assets.py --output "$HF_STAGE_DIR" \
  --release-tag vX.Y.Z --public-commit <40-character-public-commit>
python tools/ci/verify_hf_assets.py "$HF_STAGE_DIR"
python -m build --sdist --wheel
python -m twine check --strict dist/*
```
