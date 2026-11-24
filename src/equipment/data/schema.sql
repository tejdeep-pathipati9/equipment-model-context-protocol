-- Schema from PLAN.md §5.

CREATE TABLE IF NOT EXISTS employees (
    id        TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    role      TEXT NOT NULL,
    hire_date DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS equipment (
    id          INTEGER PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(id),
    item        TEXT NOT NULL,
    issued_on   DATE  -- nullable: a missing date is escalated as RECORDS_INCOMPLETE
);

CREATE TABLE IF NOT EXISTS policies (
    role           TEXT NOT NULL,
    item           TEXT NOT NULL,
    max_qty        INTEGER NOT NULL,
    refresh_months INTEGER NOT NULL,
    PRIMARY KEY (role, item)
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
