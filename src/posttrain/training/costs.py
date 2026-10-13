from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

from posttrain.common import utc_now


class CostLedger:
    """RMB ledger separate from model checkpoint; shared overhead entered once."""
    def __init__(self,path: Path):
        path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,isolation_level=None)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS entries (event_id TEXT PRIMARY KEY, run_id TEXT, category TEXT, amount_micro_cny INTEGER, evidence TEXT, recorded_at TEXT)')

    def add(self,event_id: str,run_id: str,category: str,amount_cny: str,evidence: str):
        value=Decimal(amount_cny)
        if not value.is_finite() or value<0 or not event_id or not evidence:
            raise ValueError('invalid_cost_entry')
        micro=int(value*1000000)
        row=(event_id,run_id,category,micro,evidence)
        previous=self.db.execute('SELECT event_id,run_id,category,amount_micro_cny,evidence FROM entries WHERE event_id=?',(event_id,)).fetchone()
        if previous:
            if previous!=row:
                raise ValueError('cannot_rewrite_cost_event')
            return
        self.db.execute('INSERT INTO entries VALUES (?,?,?,?,?,?)',(*row,utc_now()))

    def total(self) -> Decimal:
        micro=self.db.execute('SELECT COALESCE(SUM(amount_micro_cny),0) FROM entries').fetchone()[0]
        return Decimal(micro)/1000000

    def can_start(self,current_forecast: str,remaining_mandatory: str) -> bool:
        values=[Decimal(x) for x in (current_forecast,remaining_mandatory)]
        if any(not v.is_finite() or v<0 for v in values):
            raise ValueError('missing_complete_cost_forecast')
        return self.total()+sum(values)<=1600

    def close(self):
        self.db.close()
