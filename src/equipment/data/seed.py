"""Create and seed the SQLite database if it doesn't exist (`uv run seed`).

Synthetic data from docs/requirements.md §2 and §6, as of 2026-12-01.
"""

import os
import sqlite3
from contextlib import closing
from importlib.resources import files
from pathlib import Path

DEFAULT_DB_PATH = "./data/equipment.db"
SCHEMA_VERSION = 2  # PRAGMA user_version in schema.sql; 2 added the CHECK constraints

EMPLOYEES = [
    # id, name, role, hire_date
    ("E001", "Alice Chen", "employee", "2023-03-15"),
    ("E002", "Bob Martinez", "contractor", "2026-01-20"),
    ("E003", "Carol Singh", "employee", "2024-11-04"),
    ("E004", "Dan Okafor", "manager", "2021-06-01"),
    ("E005", "Eve Novak", "intern", "2026-06-01"),
    ("E006", "Frank Liu", "employee", "2020-02-10"),
]

EQUIPMENT = [
    # employee_id, item, issued_on
    ("E001", "monitor", "2025-06-01"),
    ("E001", "laptop", "2024-09-01"),
    ("E002", "monitor", "2026-02-01"),
    ("E003", "monitor", "2026-03-15"),
    ("E003", "monitor", "2026-03-15"),
    ("E003", "laptop", "2025-01-10"),
    ("E004", "laptop", "2025-12-01"),
    ("E004", "monitor", "2025-12-01"),
    ("E004", "monitor", "2025-12-01"),
    ("E004", "phone", "2025-12-01"),
    ("E005", "laptop", "2026-06-01"),
    ("E006", "laptop", "2023-05-01"),
]

# role, item, max_qty, refresh_months  (max_qty 0 = role not eligible for the item)
POLICIES = [
    ("intern", "laptop", 1, 36),
    ("employee", "laptop", 1, 36),
    ("manager", "laptop", 1, 24),
    ("contractor", "laptop", 0, 0),
    ("intern", "monitor", 1, 24),
    ("employee", "monitor", 2, 24),
    ("manager", "monitor", 2, 24),
    ("contractor", "monitor", 1, 24),
    ("intern", "keyboard_mouse", 1, 24),
    ("employee", "keyboard_mouse", 1, 24),
    ("manager", "keyboard_mouse", 1, 24),
    ("contractor", "keyboard_mouse", 1, 24),
    ("intern", "headset", 1, 24),
    ("employee", "headset", 1, 24),
    ("manager", "headset", 1, 24),
    ("contractor", "headset", 1, 24),
    ("intern", "phone", 0, 0),
    ("employee", "phone", 0, 0),
    ("manager", "phone", 1, 24),
    ("contractor", "phone", 0, 0),
]


def db_path() -> Path:
    return Path(os.environ.get("EQUIP_DB_PATH", DEFAULT_DB_PATH))


def create_schema(conn: sqlite3.Connection) -> None:
    schema = files("equipment.data").joinpath("schema.sql").read_text()
    conn.executescript(schema)


def is_current(conn: sqlite3.Connection) -> bool:
    """False for a database made by an older schema: CREATE TABLE IF NOT EXISTS won't update it."""
    return conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def seed_data(conn: sqlite3.Connection) -> None:
    conn.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", EMPLOYEES)
    conn.executemany(
        "INSERT INTO equipment (employee_id, item, issued_on) VALUES (?, ?, ?)", EQUIPMENT
    )
    conn.executemany("INSERT INTO policies VALUES (?, ?, ?, ?)", POLICIES)


def main() -> None:
    """Create the database once; exit with an error on one from an older schema."""
    path = db_path()
    if path.exists():
        with closing(sqlite3.connect(path)) as conn:
            current = is_current(conn)
        if not current:  # fails the container at startup instead of serving an old database
            raise SystemExit(f"seed: {path} is from an older schema; delete it and run seed again")
        print(f"seed: {path} already exists, leaving it as is")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    building = path.with_name(f"{path.name}.building")
    building.unlink(missing_ok=True)  # left by an interrupted run
    try:
        with closing(sqlite3.connect(building)) as conn, conn:  # `with conn` commits
            create_schema(conn)
            seed_data(conn)
    except BaseException:
        building.unlink(missing_ok=True)
        raise
    building.replace(path)  # atomic: the database appears complete, or not at all
    print(f"seed: created {path} ({len(EMPLOYEES)} employees, {len(EQUIPMENT)} equipment rows)")


if __name__ == "__main__":
    main()
