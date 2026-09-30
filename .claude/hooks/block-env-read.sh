#!/usr/bin/env bash
# Refuse Bash commands that would print the contents of a secret-bearing env
# file. Added 2026-09-21 after a grep whose redaction filter did not match
# printed an API key into tool output.
#
# Resolved configuration comes from the running process instead:
#   docker compose exec api python -c "from app.config import settings; print(settings.NAME)"
# while stat and ls still answer "when did it change" without exposing contents.
#
# Every segment of a compound command is inspected on its own, so a metadata
# command cannot carry a content read in beside it.
set -uo pipefail

# The file named as a path token: preceded by a separator and not part of a
# longer word such as "environment". A following "." is allowed so that
# .env.local is caught; the tracked example file is removed before matching.
SECRET_FILE_RE='(^|[^[:alnum:]_-])[.]env(rc)?([^[:alnum:]_-]|$)'

command_text=$(jq -r '.tool_input.command // ""' 2>/dev/null || printf '')

deny() {
  printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Project policy: env-file contents must not be read or printed. For a value, read it from the running process instead: docker compose exec api python -c \"from app.config import settings; print(settings.NAME)\". For modification time, list the directory (ls -la) rather than naming the file."}}'
  exit 0
}

deny_credential() {
  printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Project policy: this command could print a credential (environment dump, container config/inspect, or inline code touching a credential setting). Credentials must never reach tool output. To check which credentials are configured or whether one leaked, use: docker compose exec -T api python -m app.secret_scan (stdin, --database, --lines, --configured). It reports setting names and counts only."}}'
  exit 0
}

# Commands that print a process environment or resolved container config,
# which carry every credential. Added 2026-09-25 after a test failure printed
# the Settings object with live keys.
ENV_DUMP_RE='(^|[;&|(`[:space:]])(printenv|export[[:space:]]+-p|declare[[:space:]]+-x)([[:space:]]|$|[;&|)])|(^|[;&|(`[:space:]])env[[:space:]]*($|[;&|)])|docker[[:space:]]+(container[[:space:]]+)?inspect|docker(-|[[:space:]]+)compose([[:space:]]+-[^[:space:]]+([[:space:]]+[^-[:space:]][^[:space:]]*)?)*[[:space:]]+config|environ([^[:alnum:]_]|$)|/proc/[^[:space:]]*/environ'
# Inline code that reads a credential setting or unwraps a secret.
CREDENTIAL_CODE_RE='settings[.][A-Z0-9_]*(API_KEY|_TOKEN|SECRET_KEY|ACCESS_KEY|PASSWORD|DATABASE_URL|REDIS_URL|BROKER_URL|RESULT_BACKEND)|secret_value[[:space:]]*[(]|get_secret_value|model_dump|getattr[(][[:space:]]*settings'
[[ $command_text =~ $ENV_DUMP_RE ]] && deny_credential
[[ $command_text =~ $CREDENTIAL_CODE_RE ]] && deny_credential

probe=${command_text//.env.example/}
[[ $probe =~ $SECRET_FILE_RE ]] || exit 0

# Inspect only the segments that name the file, each on its own.
# `|| [[ -n $segment ]]` keeps the final, unterminated segment: sed's last
# line has no trailing newline, and read would otherwise discard it.
while IFS= read -r segment || [[ -n $segment ]]; do
  [[ $segment =~ $SECRET_FILE_RE ]] || continue
  verb=$(printf '%s' "$segment" | sed -E 's/^[[:space:]]+//; s/^(sudo|env)[[:space:]]+//' | awk '{print $1}')
  case "$verb" in
    stat|ls) continue ;;
    *) deny ;;
  esac
done < <(printf '%s' "$probe" | sed -E 's/(&&|\|\||;|\|)/\n/g')

exit 0
