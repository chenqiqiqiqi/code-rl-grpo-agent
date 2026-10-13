from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from posttrain.common import canonical, utc_now


def read_snapshot(path: Path) -> dict:
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        cap=db.execute('SELECT cap FROM config WHERE id=1').fetchone()[0]
        rows=db.execute('SELECT id,worst,actual,payload,created_at FROM reservations ORDER BY rowid').fetchall()
        used=sum(r[2] if r[2] is not None else r[1] for r in rows)
        return {'cap':cap,'charged_tokens':used,'remaining':cap-used,
            'reservations':[{'id':r[0],'reserved':r[1],'actual':r[2],'payload':json.loads(r[3]),'created_at':r[4]} for r in rows]}
    finally:db.close()


class BudgetExceeded(RuntimeError):
    pass


class TokenLedger:
    """Durable transactions outside checkpoints; pending work retains worst-case cost.

    Never release a reservation on timeout or after a crash without evidence of its
    actual generation. SQLite IMMEDIATE transactions serialize concurrent writers.
    """
    def __init__(self, path: Path, cap: int):
        if type(cap) is not int or cap <= 0:
            raise ValueError('cap must be a positive integer')
        path.parent.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(path,timeout=30,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), cap INTEGER NOT NULL)')
        self.db.execute('INSERT OR IGNORE INTO config VALUES (1,?)',(cap,))
        if self.db.execute('SELECT cap FROM config').fetchone()[0] != cap:
            raise ValueError('cannot_change_frozen_token_cap')
        self.cap = cap
        self.db.execute('CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, worst INTEGER NOT NULL, actual INTEGER, payload TEXT NOT NULL, created_at TEXT NOT NULL)')

    def used(self) -> int:
        return self.db.execute('SELECT COALESCE(SUM(COALESCE(actual,worst)),0) FROM reservations').fetchone()[0]

    def reserve(self, prompts: int, g: int, max_tokens: int, purpose: str) -> str:
        if any(type(v) is not int or v <= 0 for v in (prompts,g,max_tokens)):
            raise ValueError('invalid_reservation')
        worst = prompts*g*max_tokens
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.used()+worst > self.cap:
                raise BudgetExceeded('insufficient_worst_case_generation_budget')
            rid = uuid.uuid4().hex
            self.db.execute('INSERT INTO reservations VALUES (?,?,NULL,?,?)',(rid,worst,canonical({'prompts':prompts,'g':g,'max_tokens':max_tokens,'purpose':purpose}),utc_now()))
            self.db.execute('COMMIT')
            return rid
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def settle(self, reservation: str, actual_tokens: int) -> None:
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT worst,actual FROM reservations WHERE id=?',(reservation,)).fetchone()
            if not row or type(actual_tokens) is not int or not 0 <= actual_tokens <= row[0]:
                raise ValueError('invalid_settlement')
            if row[1] is not None and row[1] != actual_tokens:
                raise ValueError('cannot_rewrite_spent_tokens')
            self.db.execute('UPDATE reservations SET actual=? WHERE id=?',(actual_tokens,reservation))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def snapshot(self) -> dict:
        rows = self.db.execute('SELECT id,worst,actual,payload,created_at FROM reservations ORDER BY rowid').fetchall()
        return {'cap':self.cap,'charged_tokens':self.used(),'remaining':self.cap-self.used(),
                'reservations':[{'id':r[0],'reserved':r[1],'actual':r[2],'payload':json.loads(r[3]),'created_at':r[4]} for r in rows]}

    def close(self):
        self.db.close()
