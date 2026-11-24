"""Read access to the SQLite database: employees and their equipment."""

import sqlite3
from datetime import date

from equipment.domain.models import Employee, EquipmentUnit


class SqliteRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_employee(self, employee_id: str) -> Employee | None:
        """The employee, or None if the ID is unknown."""
        row = self._conn.execute(
            "SELECT id, name, role, hire_date FROM employees WHERE id = ?", (employee_id,)
        ).fetchone()
        if row is None:
            return None
        id_, name, role, hire_date = row
        return Employee(id=id_, name=name, role=role, hire_date=date.fromisoformat(hire_date))

    def list_equipment(self, employee_id: str) -> list[EquipmentUnit]:
        """Equipment on file for the employee, oldest first (missing dates last); may be empty."""
        rows = self._conn.execute(
            "SELECT item, issued_on FROM equipment WHERE employee_id = ? "
            "ORDER BY issued_on IS NULL, issued_on, id",
            (employee_id,),
        ).fetchall()
        return [
            EquipmentUnit(item=item, issued_on=date.fromisoformat(issued_on) if issued_on else None)
            for item, issued_on in rows
        ]
