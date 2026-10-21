"""The project command hook that keeps credentials out of tool output.

Runs `.claude/hooks/block-env-read.sh` as Claude Code does: a JSON payload on
stdin, and a deny decision on stdout or nothing. Cases from the earlier ad-hoc
tester plus the loosened `getattr(settings, ...)` rule (2026-10-11).
"""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "block-env-read.sh"
pytestmark = pytest.mark.skipif(shutil.which("jq") is None or not HOOK.exists(),
                                reason="the hook needs jq and bash")

ALLOW = [
    'cd /x && stat -f "%Sm" .env',
    'cd /x && ls -la .env',
    'cd /x && grep -c . .env.example',
    'cd /x && docker compose up -d',
    'cd /x && echo myproj.environment',
    'cd /x && python -c "import os"',
    'docker compose exec -T api python -m app.secret_scan --database',
    'docker compose logs --no-color | docker compose exec -T api python -m app.secret_scan',
    'docker compose exec api python -c "from app.config import settings; print(settings.SEARCH_POLICY_VERSION)"',
    'docker compose up -d --force-recreate api worker',
    'venv/bin/pytest -q tests/unit',
    'git diff --stat -- app/services/environment_check.py',
    'grep -rn "environment" app | head',
    # Loosened rule: a named ordinary setting.
    'python -c "from app.config import settings; print(getattr(settings, \'PROVIDER_GATE_ENABLED\', True))"',
    "python3 - <<'EOF'\nworkers = getattr(settings, \"PAPER_SOURCE_RETRIEVAL_WORKERS\", 1)\nEOF",
    'python -c "print(getattr( settings , \'ZAI_MAX_IN_FLIGHT\'))"',
]

DENY = [
    'cd /x && cat .env',
    'grep -n "KEY" .env',
    'cd /x && sed -n 1,5p .env',
    'cd /x && wc -c .env',
    'cd /x && stat .env && cat .env',
    'cd /x && git check-ignore -v .env',
    'cd /x && cat .envrc',
    'cd /x && cat app/.env',
    'cd /x && head -5 .env.local',
    'cd /x && cat "$HOME/.env"',
    'printenv',
    'env | sort',
    'docker compose exec api env',
    'docker compose exec api printenv TAVILY_API_KEY',
    'docker inspect citepet-api-api-1',
    'docker container inspect x',
    'docker compose config',
    'docker compose -f docker-compose.yml config',
    'cat /proc/1/environ',
    'python -c "import os; print(os.environ)"',
    'docker compose exec api python -c "from app.config import settings; print(settings.CORE_API_KEY)"',
    'python -c "from app.config import settings, secret_value; print(secret_value(settings.X))"',
    'python -c "from app.config import settings; print(settings.model_dump())"',
    'python -c "print(settings.DATABASE_URL)"',
    # getattr(settings, ...) still refused for an unnamed or credential-like setting.
    'python -c "print(getattr(settings, n))"',
    'python -c "for n in names: print(getattr(settings,n))"',
    'python -c "print(getattr(settings, \'ZAI_API_KEY\'))"',
    'python -c "print(getattr(settings, \\"REDIS_URL\\"))"',
    'python -c "print(getattr(settings, \'REPORT_PERSONAL_ACCESS_TOKEN\', None))"',
    'python -c "print(getattr(settings, \'CELERY_RESULT_BACKEND\'))"',
]


def _decision(command: str) -> str:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    out = subprocess.run(["bash", str(HOOK)], input=payload, capture_output=True, text=True,
                         timeout=10, cwd=HOOK.parents[2]).stdout
    return "deny" if '"permissionDecision":"deny"' in out else "allow"


@pytest.mark.parametrize("command", ALLOW)
def test_allowed(command):
    assert _decision(command) == "allow"


@pytest.mark.parametrize("command", DENY)
def test_refused(command):
    assert _decision(command) == "deny"
