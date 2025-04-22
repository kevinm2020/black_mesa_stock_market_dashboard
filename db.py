# db.py
import sqlite3
from datetime import datetime

# Create or connect to database
conn = sqlite3.connect("alerts.db", check_same_thread=False)
cursor = conn.cursor()

def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        type TEXT,
        threshold REAL,
        email TEXT,
        phone TEXT,
        status TEXT DEFAULT 'pending',
        timestamp TEXT,
        status TEXT DEFAULT 'pending'
    )
    """)
    conn.commit()

def insert_alert(ticker, alert_type, threshold, email, phone):
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO alerts (ticker, type, threshold, email, phone, timestamp,status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ticker, alert_type, threshold, email, phone, timestamp, "pending"))
    conn.commit()

def get_pending_alerts():
    cursor.execute("SELECT * FROM alerts WHERE status='pending'")
    return cursor.fetchall()

def mark_alert_triggered(alert_id):
    cursor.execute("UPDATE alerts SET status='triggered' WHERE id=?", (alert_id,))
    conn.commit()
