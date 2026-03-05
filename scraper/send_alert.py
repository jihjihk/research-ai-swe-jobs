#!/usr/bin/env python3
"""
Send scraper alerts via configured channels.

Usage:
    python send_alert.py --status success --message "Scraped 450 SWE jobs"
    python send_alert.py --status failure --message "Circuit breaker hit after 5 failures"
    python send_alert.py --status warning --message "Only 12 jobs collected (expected 200+)"

Reads alerts.conf for enabled channels and credentials.
"""

import argparse
import json
import os
import smtplib
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent  # project root (one level up from scraper/)
CONF_FILE = SCRIPT_DIR / "alerts.conf"


def load_conf() -> dict:
    """Parse alerts.conf into a dict (simple KEY=value format)."""
    conf = {}
    if not CONF_FILE.exists():
        return conf
    with open(CONF_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                # Strip surrounding quotes
                val = val.strip().strip('"').strip("'")
                conf[key.strip()] = val
    return conf


def send_email(conf: dict, subject: str, body: str):
    host = conf.get("ALERT_EMAIL_SMTP_HOST", "smtp.gmail.com")
    port = int(conf.get("ALERT_EMAIL_SMTP_PORT", "587"))
    user = conf.get("ALERT_EMAIL_SMTP_USER", "")
    passwd = conf.get("ALERT_EMAIL_SMTP_PASS", "")
    from_addr = conf.get("ALERT_EMAIL_FROM", user)
    to_addr = conf.get("ALERT_EMAIL_TO", "")

    if not all([user, passwd, to_addr]):
        print("  [email] Missing credentials, skipping")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(user, passwd)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        print(f"  [email] Sent to {to_addr}")
    except Exception as e:
        print(f"  [email] Failed: {e}")


def send_slack(conf: dict, subject: str, body: str, status: str):
    url = conf.get("ALERT_SLACK_WEBHOOK_URL", "")
    if not url:
        print("  [slack] No webhook URL, skipping")
        return

    icon = {"success": ":white_check_mark:", "failure": ":x:", "warning": ":warning:"}.get(status, ":bell:")
    payload = json.dumps({
        "text": f"{icon} *{subject}*\n```{body}```"
    }).encode()

    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15)
        print("  [slack] Sent")
    except Exception as e:
        print(f"  [slack] Failed: {e}")


def send_discord(conf: dict, subject: str, body: str, status: str):
    url = conf.get("ALERT_DISCORD_WEBHOOK_URL", "")
    if not url:
        print("  [discord] No webhook URL, skipping")
        return

    icon = {"success": "\u2705", "failure": "\u274c", "warning": "\u26a0\ufe0f"}.get(status, "\U0001f514")
    payload = json.dumps({
        "content": f"{icon} **{subject}**\n```{body}```"
    }).encode()

    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15)
        print("  [discord] Sent")
    except Exception as e:
        print(f"  [discord] Failed: {e}")


def send_ntfy(conf: dict, subject: str, body: str, status: str):
    topic = conf.get("ALERT_NTFY_TOPIC", "")
    server = conf.get("ALERT_NTFY_SERVER", "https://ntfy.sh")
    if not topic:
        print("  [ntfy] No topic, skipping")
        return

    priority = {"success": "low", "failure": "urgent", "warning": "high"}.get(status, "default")
    tags = {"success": "white_check_mark", "failure": "x", "warning": "warning"}.get(status, "bell")
    url = f"{server}/{topic}"

    try:
        req = urllib.request.Request(url, data=body.encode(), headers={
            "Title": subject,
            "Priority": priority,
            "Tags": tags,
        })
        urllib.request.urlopen(req, timeout=15)
        print("  [ntfy] Sent")
    except Exception as e:
        print(f"  [ntfy] Failed: {e}")


def send_file_alert(conf: dict, status: str, message: str, details: dict):
    path = conf.get("ALERT_FILE_PATH", str(SCRIPT_DIR / "data" / "scraper_status.json"))
    payload = {
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "details": details,
    }

    # Append to history and keep last 30 entries
    history = []
    if Path(path).exists():
        try:
            with open(path) as f:
                existing = json.load(f)
            history = existing.get("history", [])
        except (json.JSONDecodeError, KeyError):
            pass

    history.append(payload)
    history = history[-30:]

    output = {
        "current": payload,
        "history": history,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  [file] Written to {path}")


def send_macos_notification(subject: str, body: str):
    try:
        # Truncate body for notification
        short_body = body[:200].replace('"', '\\"')
        subprocess.run([
            "osascript", "-e",
            f'display notification "{short_body}" with title "{subject}"'
        ], timeout=5, capture_output=True)
        print("  [macos] Sent")
    except Exception as e:
        print(f"  [macos] Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Send scraper alerts")
    parser.add_argument("--status", required=True, choices=["success", "failure", "warning"])
    parser.add_argument("--message", required=True)
    parser.add_argument("--swe-count", type=int, default=0)
    parser.add_argument("--total-count", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()

    conf = load_conf()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"[SWE Scraper] {args.status.upper()} — {now}"
    body = (f"Status: {args.status}\n"
            f"Time: {now}\n"
            f"Message: {args.message}\n"
            f"SWE jobs: {args.swe_count}\n"
            f"Total jobs: {args.total_count}\n"
            f"Attempt: {args.attempt}")

    details = {
        "swe_count": args.swe_count,
        "total_count": args.total_count,
        "attempt": args.attempt,
    }

    print(f"Sending alert: {args.status} — {args.message}")

    if conf.get("ALERT_EMAIL_ENABLED", "").lower() == "true":
        send_email(conf, subject, body)

    if conf.get("ALERT_SLACK_ENABLED", "").lower() == "true":
        send_slack(conf, subject, body, args.status)

    if conf.get("ALERT_DISCORD_ENABLED", "").lower() == "true":
        send_discord(conf, subject, body, args.status)

    if conf.get("ALERT_NTFY_ENABLED", "").lower() == "true":
        send_ntfy(conf, subject, body, args.status)

    if conf.get("ALERT_FILE_ENABLED", "").lower() == "true":
        send_file_alert(conf, args.status, args.message, details)

    if conf.get("ALERT_MACOS_ENABLED", "").lower() == "true":
        send_macos_notification(subject, body)


if __name__ == "__main__":
    main()
