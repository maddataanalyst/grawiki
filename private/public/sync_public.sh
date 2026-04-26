#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: private/public/sync_public.sh --message "<commit message>" [--tag <tag>] [--dry-run]

Create or update the local sanitized branch from the allowlisted files at HEAD,
then optionally push it to the public remote.
EOF
}

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

info() {
    printf '%s\n' "$1"
}

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT_DIR" ]] || fail "Run this script from inside the private repository."

INCLUDE_FILE="$ROOT_DIR/private/public/include_paths.txt"
FORBIDDEN_FILE="$ROOT_DIR/private/public/forbidden_references.txt"
PRE_COMMIT_CONFIG="$ROOT_DIR/.pre-commit-config.yaml"

COMMIT_MESSAGE=""
TAG_NAME=""
DRY_RUN=0
KEEP_WORKTREE=0
TMP_ROOT=""
WORKTREE_DIR=""

read_path_list() {
    local file_path="$1"
    local -n out_ref="$2"
    local line=""

    out_ref=()
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        out_ref+=("$line")
    done < "$file_path"
}

cleanup() {
    if [[ "$KEEP_WORKTREE" -eq 0 && -n "$WORKTREE_DIR" ]]; then
        git -C "$ROOT_DIR" worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || true
    fi
    if [[ "$KEEP_WORKTREE" -eq 0 && -n "$TMP_ROOT" ]]; then
        rm -rf "$TMP_ROOT"
    fi
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --message)
            [[ $# -ge 2 ]] || fail "--message requires a value."
            COMMIT_MESSAGE="$2"
            shift 2
            ;;
        --tag)
            [[ $# -ge 2 ]] || fail "--tag requires a value."
            TAG_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

[[ -n "$COMMIT_MESSAGE" ]] || fail "Provide a curated public commit message with --message."
[[ -f "$INCLUDE_FILE" ]] || fail "Missing allowlist file: $INCLUDE_FILE"
[[ -f "$FORBIDDEN_FILE" ]] || fail "Missing forbidden-reference file: $FORBIDDEN_FILE"

git -C "$ROOT_DIR" remote get-url public >/dev/null 2>&1 || fail "Missing required public remote."

if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=all)" ]]; then
    fail "Public sync requires a completely clean private working tree."
fi

if [[ -n "$TAG_NAME" ]]; then
    if git -C "$ROOT_DIR" rev-parse --verify --quiet "refs/tags/$TAG_NAME" >/dev/null; then
        fail "Local tag $TAG_NAME already exists."
    fi
    if git -C "$ROOT_DIR" ls-remote --exit-code --tags public "refs/tags/$TAG_NAME" >/dev/null 2>&1; then
        fail "Tag $TAG_NAME already exists on the public remote."
    fi
fi

declare -a INCLUDE_PATHS
declare -a FORBIDDEN_PATTERNS

read_path_list "$INCLUDE_FILE" INCLUDE_PATHS
read_path_list "$FORBIDDEN_FILE" FORBIDDEN_PATTERNS

[[ ${#INCLUDE_PATHS[@]} -gt 0 ]] || fail "The allowlist is empty."
[[ ${#FORBIDDEN_PATTERNS[@]} -gt 0 ]] || fail "The forbidden-reference list is empty."

PUBLIC_BASE_REF=""
if git -C "$ROOT_DIR" ls-remote --exit-code --heads public main >/dev/null 2>&1; then
    git -C "$ROOT_DIR" fetch public main:refs/remotes/public/main >/dev/null
    PUBLIC_BASE_REF="refs/remotes/public/main"
elif git -C "$ROOT_DIR" show-ref --verify --quiet refs/heads/sanitized; then
    PUBLIC_BASE_REF="refs/heads/sanitized"
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/grawiki-public.XXXXXX")"
WORKTREE_DIR="$TMP_ROOT/sanitized-worktree"

if [[ -n "$PUBLIC_BASE_REF" ]]; then
    git -C "$ROOT_DIR" worktree add --detach "$WORKTREE_DIR" "$PUBLIC_BASE_REF" >/dev/null
    git -C "$WORKTREE_DIR" switch -C sanitized >/dev/null
else
    git -C "$ROOT_DIR" worktree add --detach "$WORKTREE_DIR" HEAD >/dev/null
    git -C "$WORKTREE_DIR" switch --orphan sanitized >/dev/null
fi

git -C "$WORKTREE_DIR" rm -r --quiet --ignore-unmatch . >/dev/null 2>&1 || true
git -C "$WORKTREE_DIR" clean -fdx >/dev/null

git -C "$ROOT_DIR" archive --format=tar HEAD "${INCLUDE_PATHS[@]}" | tar -xf - -C "$WORKTREE_DIR"
git -C "$WORKTREE_DIR" add --all

if git -C "$WORKTREE_DIR" diff --cached --quiet; then
    info "No public changes to sync."
    exit 0
fi

EXPECTED_FILE_LIST="$TMP_ROOT/expected_paths.txt"
ACTUAL_FILE_LIST="$TMP_ROOT/actual_paths.txt"

git -C "$ROOT_DIR" ls-tree -r --name-only HEAD -- "${INCLUDE_PATHS[@]}" | LC_ALL=C sort -u > "$EXPECTED_FILE_LIST"
git -C "$WORKTREE_DIR" ls-files | LC_ALL=C sort -u > "$ACTUAL_FILE_LIST"

if ! diff -u "$EXPECTED_FILE_LIST" "$ACTUAL_FILE_LIST" >/dev/null; then
    diff -u "$EXPECTED_FILE_LIST" "$ACTUAL_FILE_LIST" || true
    fail "Sanitized tree does not exactly match the allowlisted export."
fi

FOUND_FORBIDDEN_REFERENCE=0
for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    if git -C "$WORKTREE_DIR" grep -n -I -F -e "$pattern" -- . >/dev/null 2>&1; then
        printf 'Forbidden reference found for pattern: %s\n' "$pattern" >&2
        git -C "$WORKTREE_DIR" grep -n -I -F -e "$pattern" -- . || true
        FOUND_FORBIDDEN_REFERENCE=1
    fi
done

[[ "$FOUND_FORBIDDEN_REFERENCE" -eq 0 ]] || fail "Forbidden references were found in the mirrored content."

uv run --directory "$WORKTREE_DIR" pre-commit run --config "$PRE_COMMIT_CONFIG" gitleaks --all-files

if [[ "$DRY_RUN" -eq 1 ]]; then
    KEEP_WORKTREE=1
    info "Dry run complete. Inspect the staged sanitized worktree at: $WORKTREE_DIR"
    git -C "$WORKTREE_DIR" status --short
    git -C "$WORKTREE_DIR" diff --cached --stat
    exit 0
fi

git -C "$WORKTREE_DIR" commit -m "$COMMIT_MESSAGE"

if [[ -n "$TAG_NAME" ]]; then
    git -C "$WORKTREE_DIR" tag -a "$TAG_NAME" -m "Public release $TAG_NAME"
    git -C "$WORKTREE_DIR" push public sanitized:main "refs/tags/$TAG_NAME"
else
    git -C "$WORKTREE_DIR" push public sanitized:main
fi

info "Public sync completed successfully."
