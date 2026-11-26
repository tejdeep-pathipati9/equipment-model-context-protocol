-- Schema from PLAN.md §5. CHECK constraints refuse bad values when they are written, so the
-- tools never read them. `date(x) IS x` holds only for a real YYYY-MM-DD date.

-- Raise with every schema change: a database made by an older schema is refused (seed.py).
PRAGMA user_version = 2;

CREATE TABLE IF NOT EXISTS employees (
    id        TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    role      TEXT NOT NULL,
    hire_date DATE NOT NULL CHECK (date(hire_date) IS hire_date)
);

CREATE TABLE IF NOT EXISTS equipment (
    id          INTEGER PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(id),
    item        TEXT NOT NULL,
    issued_on   DATE  -- nullable: a missing date is escalated as RECORDS_INCOMPLETE
                CHECK (issued_on IS NULL OR date(issued_on) IS issued_on)
);

CREATE TABLE IF NOT EXISTS policies (
    role           TEXT NOT NULL,
    item           TEXT NOT NULL,
    max_qty        INTEGER NOT NULL CHECK (typeof(max_qty) = 'integer' AND max_qty >= 0),
    refresh_months INTEGER NOT NULL CHECK (typeof(refresh_months) = 'integer'),
    PRIMARY KEY (role, item),
    CHECK (refresh_months > 0 OR (max_qty = 0 AND refresh_months = 0))  -- 0 only if not eligible
);

-- employee_id is deliberately not a foreign key: unknown employees can still be escalated.
CREATE TABLE IF NOT EXISTS escalations (
    id          TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL,
    request     TEXT NOT NULL,
    reason      TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    created_at  TIMESTAMP NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open'
);
