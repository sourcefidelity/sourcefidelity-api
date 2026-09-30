"""Find configured credentials in logs, files or the database without revealing them.

Run inside the application container, where the credentials are already
configured, so no credential ever crosses into the caller's terminal:

    docker compose logs --no-color | docker compose exec -T api python -m app.secret_scan
    git diff --cached | docker compose exec -T api python -m app.secret_scan
    docker compose exec -T api python -m app.secret_scan --database
    docker compose exec -T api python -m app.secret_scan --configured   # names only
    docker compose exec -T api python -m app.secret_scan --lines < session.jsonl

Output names the setting and a count, never the value or any excerpt around
it. Exit status 1 means at least one credential was found. Any internal error
is reported by exception class only, because a traceback can carry locals.
"""
from __future__ import annotations

import base64
import json
import sys
from urllib.parse import quote

from pydantic import SecretStr

from app.config import Settings, secret_value, settings

_MIN_LENGTH = 8   # shorter values match ordinary text and are not credentials


def configured_secrets() -> dict[str, str]:
    """Credential values that differ from the public development defaults."""
    found = {}
    for name, field in Settings.model_fields.items():
        if field.annotation not in (SecretStr, SecretStr | None):
            continue
        value = secret_value(getattr(settings, name))
        default = secret_value(field.default) if isinstance(field.default, SecretStr) else None
        if value and value.strip() and value != default and len(value.strip()) >= _MIN_LENGTH:
            found[name] = value.strip()
    return found


def _forms(value: str) -> set[bytes]:
    """The value as written, URL-encoded, and base64-encoded (Basic auth, JSON dumps)."""
    raw = value.encode()
    forms = {raw, quote(value, safe="").encode(), base64.b64encode(raw).rstrip(b"=")}
    return {form for form in forms if len(form) >= _MIN_LENGTH}


def scan_bytes(data: bytes, secrets: dict[str, str]) -> dict[str, int]:
    hits = {}
    for name, value in secrets.items():
        count = sum(data.count(form) for form in _forms(value))
        if count:
            hits[name] = count
    return hits


def scan_database(secrets: dict[str, str]) -> dict[str, int]:
    from sqlalchemy import text

    from app.database import engine

    hits: dict[str, int] = {}
    with engine.connect() as connection:
        tables = [row[0] for row in connection.execute(text(
            "SELECT quote_ident(table_schema) || '.' || quote_ident(table_name) "
            "FROM information_schema.tables WHERE table_type = 'BASE TABLE' "
            "AND table_schema NOT IN ('pg_catalog', 'information_schema')"))]
        for table in tables:
            for name, value in secrets.items():
                for form in _forms(value):
                    count = connection.execute(
                        text(f"SELECT count(*) FROM {table} AS t WHERE strpos(t::text, :needle) > 0"),
                        {"needle": form.decode()}).scalar()
                    if count:
                        hits[name] = hits.get(name, 0) + count
    return hits


def main(argv: list[str]) -> int:
    try:
        secrets = configured_secrets()
        if "--configured" in argv:
            print(json.dumps({"configured": sorted(secrets)}))
            return 0
        if "--database" in argv:
            hits, scanned = scan_database(secrets), "database"
        elif "--lines" in argv:
            # Locate hits in JSON-lines input by line number and record
            # metadata only (timestamp, type), never the line's content.
            data = sys.stdin.buffer.read()
            hits, scanned, located = {}, f"{len(data)} bytes", []
            for number, line in enumerate(data.splitlines(), 1):
                line_hits = scan_bytes(line, secrets)
                if not line_hits:
                    continue
                for name, count in line_hits.items():
                    hits[name] = hits.get(name, 0) + count
                try:
                    record = json.loads(line)
                    meta = {k: record.get(k) for k in ("timestamp", "type", "sessionId") if isinstance(record, dict)}
                except ValueError:
                    meta = {}
                located.append({"line": number, "names": sorted(line_hits), **meta})
            print(json.dumps({"located": located}))
        else:
            data = sys.stdin.buffer.read()
            hits, scanned = scan_bytes(data, secrets), f"{len(data)} bytes"
    except Exception as exc:   # never print a traceback: it can carry locals
        print(json.dumps({"error": type(exc).__name__}))
        return 2
    print(json.dumps({"scanned": scanned, "credentials_checked": sorted(secrets),
                      "found": hits}, sort_keys=True))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
