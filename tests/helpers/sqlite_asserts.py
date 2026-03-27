import sqlite3
from pathlib import Path


def create_responses_table(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE responses (
            input_hash TEXT NOT NULL,
            task_name TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            model TEXT NOT NULL,
            response_json TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            tokens_used INTEGER,
            PRIMARY KEY (input_hash, task_name, prompt_version)
        )
        """
    )
    conn.commit()
    return conn
