"""SQLite 持久化层：对话原文（messages 表）与空闲摘要（instructions 表）的读写。

分工约定：
- messages：每轮问答实时写入，一行一条消息，只增不改
- instructions：每个 (user_name, session_id) 只有一行摘要，空闲 10 分钟后 upsert 覆盖
"""
import sqlite3
import datetime
from pathlib import Path
from contextlib import closing

DB_PATH = Path(__file__).parent / 'chat_history.db'


def _conn():
    """新建数据库连接并开启 WAL 模式（读写不互斥，降低锁冲突）。"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA journal_mode = WAL;')
    return conn


def init_db():
    """建表与索引（幂等）：messages 存对话原文，instructions 存每会话一条的摘要。"""
    with closing(_conn()) as conn:
        with conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_session ON messages(user_name,session_id)")
            conn.execute("""CREATE TABLE IF NOT EXISTS instructions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                instruction TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_name,session_id))""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_instructions_user_session ON instructions(user_name,session_id)")


def save_message(user_name: str, session_id: str, role: str, content: str):
    """写入一条对话原文（role 为 user 或 assistant），时间戳精确到秒。"""
    now_time = datetime.datetime.now().isoformat(timespec='seconds')
    with closing(_conn()) as conn:
        with conn:
            conn.execute("INSERT INTO messages(user_name,session_id,role,content,created_at) VALUES(?,?,?,?,?)",
                         (user_name, session_id, role, content, now_time))


def save_instruction(user_name: str, session_id: str, instruction: str):
    """UPSERT 会话摘要：该 (user_name, session_id) 已有摘要则覆盖，没有则插入。"""
    with closing(_conn()) as conn:
        now_time = datetime.datetime.now().isoformat(timespec='seconds')
        with conn:
            conn.execute("""INSERT INTO instructions(user_name,session_id,instruction,created_at) VALUES(?,?,?,?)
            ON CONFLICT(user_name,session_id) DO UPDATE SET instruction = excluded.instruction,created_at = excluded.created_at"""
                         , (user_name, session_id, instruction, now_time))

def get_messages(user_name:str,session_id:str):
    """查询会话所有消息（按时间顺序）。"""
    with closing(_conn()) as conn:
        with conn:
            rows = conn.execute("""SELECT * FROM (
    SELECT * FROM messages WHERE user_name = ? AND session_id = ? ORDER BY id DESC LIMIT 10 ) ORDER BY id ASC
    """,(user_name,session_id)).fetchall()
    return rows

def get_instruction(user_name:str,session_id:str):
    """查询会话摘要（若存在）。"""
    with closing(_conn()) as conn:
        with conn:
            row = conn.execute("""SELECT * FROM instructions WHERE user_name = ? AND session_id = ?
            """,(user_name,session_id)).fetchone()
    return row