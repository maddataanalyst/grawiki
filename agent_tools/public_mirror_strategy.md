# Public Mirror Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Maintain a public-facing GitHub repository that mirrors a *curated subset* of this private project, updated on demand via a single well-described commit per sync. Without maintaining a second working copy or doing manual file-by-file copying.

**Architecture:** One local repository, two git remotes (`origin` → private, `public` → public). A small shell script builds each public sync by checking out the public tip in a temporary worktree, wiping its tree, copying the allowed files from the current private working tree (filtered through a `.publicignore` list), making one commit with a user-supplied message, and pushing to `public/main`. Each sync is exactly one commit on the public side; private history never leaks because the script never pushes private commits to the public remote.

**Tech Stack:** `git` (worktrees, remotes, orphan branches), `rsync` for filtered file copies, a single shell script — no new Python dependencies.

---

## Constraints & Requirements

1. **No history leakage.** Private commit messages, branch names, author emails, SHAs must not appear on the public remote.
2. **Curated commits.** Each public sync is one commit with a message the developer writes for that sync (release-notes style). Private fine-grained history is intentionally flattened.
3. **File-level filtering.** Some tracked files must never appear publicly: `AGENTS.md`, `agent_tools/TODOS.md`, `notebooks/` (including `local_falkor.db`, `debug.py`, `debug.ipynb`, `experimental_data/`), `agent_tools/similar_entities_usage_scenarios.md`, `agent_tools/extraction_speedup_plan.md`, `agent_tools/public_mirror_strategy.md` (these three plan docs are internal planning, not user-facing docs), `.claude/` if it exists, any `.env*`, `.venv/`, `__pycache__/`.
4. **Incremental.** The developer syncs "occasionally" — once every few commits to weeks. The workflow must be fast enough to run ad-hoc (`./scripts/sync_public.sh "Add vector entity similarity"`).
5. **Safe defaults.** The script must never force-push over history or touch `origin`. It only touches `public`.
6. **No second working copy.** The worktree used during sync is ephemeral (created and removed by the script).

---

## Approach Comparison

| Option | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| A. `git filter-repo` on a separate clone, rewrite history each sync | Preserves fine-grained history | Heavy, re-rewrites every sync, fragile when exclude list changes | ❌ too heavy |
| B. Two branches in one repo + `git subtree` | Built-in git, no script | Subtree is for subdirectories, not arbitrary file excludes | ❌ wrong tool |
| C. Orphan-branch + rsync script, one commit per sync | Simple, no history leakage, easy to describe commits | Loses fine-grained public history (by design) | ✅ best fit |
| D. Manual: maintain two working copies and `cp -r` each time | Maximum control | Exactly what the user said they don't want | ❌ rejected |

Option C matches every requirement and is ~50 lines of shell. Plans below implement it.

---

## File Structure

- **Create** `.publicignore` — `rsync`-format exclude list (one pattern per line, no comments required but supported with `#`).
- **Create** `scripts/sync_public.sh` — the sync script. Takes one argument (the public commit message) or opens `$EDITOR` for multi-line input.
- **Create** `scripts/public_first_sync.sh` — one-off helper for the initial public push (different because there is no `public/main` to base on).
- **Modify** `AGENTS.md` — add a short section describing the public-sync workflow so future agents know the convention. (Note: `AGENTS.md` itself stays private per the exclude list.)
- **No Python changes.** No test changes.

---

## Progress

| Task | Status | Notes |
| --- | --- | --- |
| 1. Define exclude list | ⏳ Not started | |
| 2. Write `scripts/sync_public.sh` | ⏳ Not started | Depends on 1 |
| 3. Write `scripts/public_first_sync.sh` | ⏳ Not started | Depends on 1, 2 |
| 4. One-time setup: create public repo on GitHub, add `public` remote | ⏳ Not started | Manual GitHub action |
| 5. First sync (smoke test) | ⏳ Not started | Depends on 4 |
| 6. Document workflow in `AGENTS.md` | ⏳ Not started | |

---

## Task 1: Define the exclude list

**Files:**
- Create: `.publicignore`

- [ ] **Step 1: Write the initial exclude list**

Create `.publicignore` at the repo root with the following (adjust if some items shouldn't actually be excluded — revisit before first push):

```
# Internal planning and agent instructions
AGENTS.md
CLAUDE.md
agent_tools/TODOS.md
agent_tools/similar_entities_usage_scenarios.md
agent_tools/extraction_speedup_plan.md
agent_tools/public_mirror_strategy.md

# Exploratory / debug material
notebooks/
scripts/sync_public.sh
scripts/public_first_sync.sh
.publicignore

# Local state / secrets / caches
.env
.env.*
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.mypy_cache/
.claude/
.DS_Store

# Git metadata (rsync should skip this by default, but belt-and-braces)
.git/
```

Notes on the "exclude the scripts themselves" line: the sync scripts are part of the private workflow, not something a public consumer should see. Same for `.publicignore`. These live in the private repo only.

- [ ] **Step 2: Sanity-check the list**

Run:
```bash
rsync -avn --exclude-from=.publicignore --exclude='.git/' ./ /tmp/publicsync-preview/
```

Read the output carefully. Anything you see there is what will land on the public mirror. If something surprising is in the list (API key, local DB file, personal note), add it to `.publicignore` and re-run.

- [ ] **Step 3: Commit**

```bash
git add .publicignore
git commit -m "chore: add public-mirror exclude list"
```

---

## Task 2: Write the sync script

**Files:**
- Create: `scripts/sync_public.sh`

- [ ] **Step 1: Write the script**

Create `scripts/sync_public.sh` (make executable afterwards with `chmod +x`):

```bash
#!/usr/bin/env bash
# Sync the current private working tree to the public remote as ONE curated commit.
#
# Usage:
#   scripts/sync_public.sh "commit message"
#   scripts/sync_public.sh                  # opens $EDITOR for multi-line message
#
# Requirements:
#   - `public` git remote pointing to the public GitHub repo
#   - Current branch is clean (no uncommitted changes) — the script refuses otherwise
#   - `.publicignore` at the repo root

set -euo pipefail

PUBLIC_REMOTE="${PUBLIC_REMOTE:-public}"
PUBLIC_BRANCH="${PUBLIC_BRANCH:-main}"
EXCLUDE_FILE="${EXCLUDE_FILE:-.publicignore}"

PRIVATE_ROOT="$(git rev-parse --show-toplevel)"
cd "$PRIVATE_ROOT"

# Refuse if working tree is dirty — we only mirror committed state.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree has uncommitted changes. Commit or stash first." >&2
  exit 1
fi

# Collect the commit message.
if [[ $# -ge 1 ]]; then
  COMMIT_MSG="$1"
else
  TMP_MSG_FILE="$(mktemp)"
  trap 'rm -f "$TMP_MSG_FILE"' EXIT
  printf '# Enter the public-sync commit message. Lines starting with # are ignored.\n' >"$TMP_MSG_FILE"
  "${EDITOR:-vi}" "$TMP_MSG_FILE"
  COMMIT_MSG="$(grep -v '^#' "$TMP_MSG_FILE" | sed '/^$/N;/^\n$/D')"
  if [[ -z "$COMMIT_MSG" ]]; then
    echo "error: empty commit message." >&2
    exit 1
  fi
fi

# Fetch the public tip so our commit stacks on top of previous syncs.
git fetch "$PUBLIC_REMOTE" "$PUBLIC_BRANCH"

# Ephemeral worktree based on public/main.
WORKTREE_DIR="$(mktemp -d)"
trap 'git worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || true; rm -rf "$WORKTREE_DIR"' EXIT

git worktree add --detach "$WORKTREE_DIR" "$PUBLIC_REMOTE/$PUBLIC_BRANCH"

# Wipe the worktree tree (keeping .git metadata), then rsync the private state in.
pushd "$WORKTREE_DIR" >/dev/null
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
popd >/dev/null

rsync -a --delete \
  --exclude='.git/' \
  --exclude-from="$PRIVATE_ROOT/$EXCLUDE_FILE" \
  "$PRIVATE_ROOT/" \
  "$WORKTREE_DIR/"

# Commit and push. No author rewrite is needed — the developer's own name/email
# will appear publicly, same as any normal commit.
pushd "$WORKTREE_DIR" >/dev/null
git add -A
if git diff --cached --quiet; then
  echo "info: no changes to sync."
  exit 0
fi
git commit -m "$COMMIT_MSG"
git push "$PUBLIC_REMOTE" HEAD:"$PUBLIC_BRANCH"
popd >/dev/null

echo "done: pushed to $PUBLIC_REMOTE/$PUBLIC_BRANCH"
```

- [ ] **Step 2: Make it executable and commit**

```bash
chmod +x scripts/sync_public.sh
git add scripts/sync_public.sh
git commit -m "chore: add public-mirror sync script"
```

- [ ] **Step 3: Dry-run the script locally (no push)**

Temporarily replace the `git push` line with `echo "would push"` and run:

```bash
./scripts/sync_public.sh "test: initial sync"
```

Inspect the temporary worktree in `/tmp/*` before it's cleaned up (comment out the `trap` for the first run). Confirm the excluded files are absent and the included files are present.

Revert the script changes before committing.

---

## Task 3: Write the first-sync helper

**Files:**
- Create: `scripts/public_first_sync.sh`

- [ ] **Step 1: Write the script**

The regular script requires `public/main` to exist. The first sync needs an orphan branch instead:

```bash
#!/usr/bin/env bash
# One-time bootstrap: create the initial commit on the public remote.
# After this runs successfully, use scripts/sync_public.sh for all subsequent syncs.

set -euo pipefail

PUBLIC_REMOTE="${PUBLIC_REMOTE:-public}"
PUBLIC_BRANCH="${PUBLIC_BRANCH:-main}"
EXCLUDE_FILE="${EXCLUDE_FILE:-.publicignore}"

PRIVATE_ROOT="$(git rev-parse --show-toplevel)"
cd "$PRIVATE_ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree has uncommitted changes." >&2
  exit 1
fi

COMMIT_MSG="${1:-Initial public release}"

WORKTREE_DIR="$(mktemp -d)"
trap 'git worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || true; rm -rf "$WORKTREE_DIR"' EXIT

# Base on current HEAD just to have a starting tree; we will orphan-branch it.
git worktree add --detach "$WORKTREE_DIR" HEAD

pushd "$WORKTREE_DIR" >/dev/null
git checkout --orphan public-initial

# Wipe the tree, then rsync.
git rm -rf --cached . >/dev/null 2>&1 || true
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
popd >/dev/null

rsync -a \
  --exclude='.git/' \
  --exclude-from="$PRIVATE_ROOT/$EXCLUDE_FILE" \
  "$PRIVATE_ROOT/" \
  "$WORKTREE_DIR/"

pushd "$WORKTREE_DIR" >/dev/null
git add -A
git commit -m "$COMMIT_MSG"
git push "$PUBLIC_REMOTE" HEAD:"$PUBLIC_BRANCH"
popd >/dev/null

echo "done: initial public push complete"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/public_first_sync.sh
git add scripts/public_first_sync.sh
git commit -m "chore: add first-sync bootstrap script"
```

---

## Task 4: One-time setup on GitHub

This task is **manual** — it cannot be automated from the repo.

- [ ] **Step 1: Create the public GitHub repo**

Use the GitHub UI or `gh` CLI. Name it whatever you want (e.g. `grawiki`). Create it empty — no README, no license, no `.gitignore` — otherwise the first force-push will conflict.

If you picked a license, add a `LICENSE` file to the private repo so it appears in the public sync instead of being auto-added by GitHub.

- [ ] **Step 2: Add the remote locally**

```bash
git remote add public git@github.com:<your-username>/grawiki.git
git remote -v  # verify: origin=private, public=public
```

- [ ] **Step 3: Confirm the exclude list matches reality**

Preview what will land:

```bash
mkdir -p /tmp/publicsync-preview
rsync -avn --delete --exclude='.git/' --exclude-from=.publicignore ./ /tmp/publicsync-preview/ | less
```

Add missing excludes or remove over-eager ones before running the first sync.

---

## Task 5: First sync (smoke test)

- [ ] **Step 1: Run the bootstrap**

```bash
./scripts/public_first_sync.sh "Initial public release of grawiki"
```

- [ ] **Step 2: Verify on GitHub**

Open the public repo in a browser and inspect the tree. Confirm:
- `AGENTS.md` is **absent**.
- `notebooks/` is **absent**.
- `agent_tools/TODOS.md` and the three plan docs are **absent**.
- `src/grawiki/` is **present**.
- `tests/` is **present**.
- `agent_tools/CODEMAP.md`, `README.md`, `pyproject.toml` are **present**.
- The commit log shows exactly **one commit** with your message.

- [ ] **Step 3: Run a second sync immediately**

Make a trivial change privately (e.g. fix a typo in `README.md` and commit), then:

```bash
./scripts/sync_public.sh "Fix typo in README"
```

Confirm on GitHub that the public log now has **two commits**, with the linear parent chain intact. Private-side SHAs should not appear.

---

## Task 6: Document the workflow

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add a section**

Append to `AGENTS.md`:

```markdown
## Public Mirror

This repository is private. A curated public mirror is maintained via
`scripts/sync_public.sh`. Each sync produces a single well-described commit on
the `public/main` branch. Files excluded from the public mirror are listed in
`.publicignore`.

**When to sync:** after shipping a coherent change worth showing externally
(feature, bug fix, doc improvement). Do not sync in-progress work.

**How:** `./scripts/sync_public.sh "short release-note-style message"` from
a clean working tree. The script refuses to run with uncommitted changes.

**Never:** push private branches to the `public` remote directly, and never
add tracked files containing secrets or internal planning without updating
`.publicignore` in the same commit.
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: document public-mirror workflow"
```

Note: this commit will itself be excluded from the next public sync because `AGENTS.md` is on the exclude list. That is intentional.

---

## Risks and Rollback Notes

- **Accidental leakage.** The exclude list is a safety net, not a secret scanner. Before every sync, run the dry-run preview (Task 4 Step 3). If a secret ever does leak, rotate it immediately and rewrite the public branch (`git push public --force HEAD:main` from a corrected worktree).
- **Force-push to public.** The regular sync script is *not* a force push — each sync is a fast-forward commit. Only the bootstrap (`public_first_sync.sh`) does an implicit force push into a freshly-created repo. If you ever need to rewrite public history after a leak, do it deliberately with an explicit `--force` flag you add manually.
- **Author identity.** Commits on the public remote will carry your configured `user.name` and `user.email`. If those are an internal email, change them before the first sync: `git config user.email "public@example.com"` in a `.git/config` scoped to the worktree (or set globally).
- **Public consumers expect a LICENSE.** Add one to the private repo (`LICENSE` at root) so it flows into the public mirror. Without a license, the code is technically not redistributable.
- **Someone else's PR to the public repo** cannot be merged back into private directly — you would need to cherry-pick the patch by hand into the private `main`. This is an accepted limitation of Option C; flag it in `AGENTS.md` if contributions become a real concern.
- **Script stays private.** `scripts/sync_public.sh` is on the exclude list, so a public consumer will not see it. If you ever want to disclose the mirroring approach for transparency, publish a description in `README.md` instead of the script itself.
