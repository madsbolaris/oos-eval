#!/usr/bin/env bash
set -euo pipefail

repo="madsbolaris/oos-eval"
environment="kev-training"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: GitHub CLI (gh) is not installed" >&2
  exit 1
fi

gh auth status >/dev/null

read -r -s -p "Modal token ID: " modal_token_id
printf '\n'
read -r -s -p "Modal token secret: " modal_token_secret
printf '\n'

if [[ -z "$modal_token_id" || -z "$modal_token_secret" ]]; then
  echo "error: both values are required" >&2
  exit 1
fi

printf '%s' "$modal_token_id" |
  gh secret set MODAL_TOKEN_ID --repo "$repo" --env "$environment"
printf '%s' "$modal_token_secret" |
  gh secret set MODAL_TOKEN_SECRET --repo "$repo" --env "$environment"

unset modal_token_id modal_token_secret

echo "Configured environment secrets:"
gh secret list --repo "$repo" --env "$environment"