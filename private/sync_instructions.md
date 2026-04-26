# Public Sync Instructions

Use untagged syncs for normal public updates. Use tagged syncs only for releases.

## Sync without a tag

Use this for normal public updates such as code changes, docs updates, and intermediate milestones that should not publish to PyPI.

Command:

```bash
private/public/sync_public.sh --message "feat: improve retrieval flow"
```

What it does:

- rebuilds `sanitized` from private `HEAD`
- creates one curated public commit
- pushes `sanitized` to `public/main`
- does not create a tag
- does not trigger PyPI publishing

## Sync with a tag

Use this for releases that should be published from the public repository.

Command:

```bash
private/public/sync_public.sh --message "release: 0.1.0" --tag v0.1.0
```

What it does:

- creates the curated public commit
- creates an annotated tag such as `v0.1.0`
- pushes the commit and tag to the public repository
- triggers `.github/workflows/publish.yml`

## Should every sync have a tag?

No.

Recommended rule:

- do not use tags for routine syncs
- use tags only for releases

## Typical workflow

Normal public update:

```bash
private/public/sync_public.sh --message "docs: clarify notebook setup"
```

Release:

1. Update the version in `pyproject.toml`.
2. Run validation locally.
3. Sync with a tag.

```bash
private/public/sync_public.sh --message "release: 0.1.1" --tag v0.1.1
```

## Current script caveat

`sync_public.sh` only creates a tag when the sync also creates a new public commit.

That means:

- a tagged release sync works when there are new changes to publish
- the script exits early if there is nothing new to sync, even if `--tag` is provided

Recommended practice:

- use untagged syncs for normal updates
- use a tag on the same sync that publishes a release
- avoid syncing first and trying to add the release tag later with the script

If a later tag is needed, create it manually:

```bash
git fetch public
git branch -f sanitized public/main
git tag -a v0.1.1 sanitized -m "Public release v0.1.1"
git push public refs/tags/v0.1.1
```
