"""Create the SQLite database if it doesn't exist (`uv run seed`).

Currently creates the schema only; seed rows are added in build step 4.
"""

import os
import sqlite3
from importlib.resources import files
from pathlib import Path

DEFAULT_DB_PATH = "./data/equipment.db"


def db_path() -> Path:
    return Path(os.environ.get("EQUIP_DB_PATH", DEFAULT_DB_PATH))


def create_schema(conn: sqlite3.Connection) -> None:
    schema = files("equipment.data").joinpath("schema.sql").read_text()
    conn.executescript(schema)


def main() -> None:
    path = db_path()
    if path.exists():
        print(f"seed: {path} already exists, leaving it as is")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        create_schema(conn)
    print(f"seed: created {path}")


if __name__ == "__main__":
    main()
