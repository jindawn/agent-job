"""Demo business database (SQLite). Seeded on first use."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY, customer TEXT, phone TEXT, status TEXT,
  amount REAL, created_at TEXT, logistics_no TEXT
);
CREATE TABLE IF NOT EXISTS tickets (
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, description TEXT,
  priority TEXT, status TEXT DEFAULT 'open', created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

SEED_ORDERS = [
    ("EC20250001", "张伟", "13912345678", "已签收", 258.00, "2025-03-02", "SF1001"),
    ("EC20250002", "李娜", "13812345678", "已发货", 99.00, "2025-03-05", "SF1002"),
    ("EC20250003", "王芳", "13700001111", "已发货", 1299.00, "2025-03-06", "YT2003"),
    ("EC20250004", "赵磊", "15012349876", "待发货", 45.50, "2025-03-07", None),
    ("EC20250005", "陈静", "18611112222", "已退款", 328.00, "2025-02-20", "SF1005"),
    ("EC20250006", "刘洋", "17788889999", "已签收", 76.00, "2025-03-01", "ZT3006"),
    ("EC20250007", "周敏", "13566667777", "待发货", 560.00, "2025-03-08", None),
    ("EC20250008", "吴强", "13399990000", "已签收", 89.90, "2025-02-27", "YT2008"),
]


def init_db(path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
            conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", SEED_ORDERS)
