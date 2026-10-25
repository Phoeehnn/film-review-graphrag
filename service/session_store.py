"""SQLite optimistic commits, short transactions and same-day session expiry."""
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .session_models import SessionState


class SessionConflict(RuntimeError):
    pass


class SessionStore:
    def __init__(self,directory,clock=time.time):
        directory.mkdir(parents=True,exist_ok=True)
        self.path=directory/'sessions.sqlite3'
        self.clock=clock
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, version INTEGER NOT NULL, expires REAL NOT NULL, body TEXT NOT NULL)')

    @contextmanager
    def connection(self):
        db=sqlite3.connect(self.path,timeout=5)
        try:
            with db:yield db
        finally:db.close()

    def create(self):
        now=self.clock()
        local=datetime.fromtimestamp(now,timezone(timedelta(hours=8)))
        expiry=(local+timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
        state=SessionState(id=str(uuid4()),created_at=now,expires_at=expiry)
        with self.connection() as db:
            db.execute('DELETE FROM sessions WHERE expires<=?',(now,))
            db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(state.id,state.version,expiry,state.model_dump_json()))
        return state

    def load(self,identifier):
        with self.connection() as db:
            row=db.execute('SELECT body,expires FROM sessions WHERE id=?',(identifier,)).fetchone()
        if not row or row[1]<=self.clock():raise KeyError('会话不存在或已过期，请创建新会话')
        return SessionState.model_validate_json(row[0])

    def commit(self,state,expected_version):
        new=state.model_copy(update={'version':expected_version+1})
        with self.connection() as db:
            cursor=db.execute('UPDATE sessions SET version=?,body=? WHERE id=? AND version=? AND expires>?',
                (new.version,new.model_dump_json(),new.id,expected_version,self.clock()))
            if cursor.rowcount!=1:raise SessionConflict('会话已被其他请求更新或已过期，请刷新后重试')
        return new

    def delete(self,identifier):
        with self.connection() as db:db.execute('DELETE FROM sessions WHERE id=?',(identifier,))
