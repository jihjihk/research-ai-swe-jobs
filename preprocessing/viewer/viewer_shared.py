from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .viewer-kicker {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.78rem;
            color: #6b7280;
            margin-bottom: 0.15rem;
        }
        .viewer-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .viewer-subtitle {
            color: #4b5563;
            max-width: 66rem;
            margin-bottom: 1rem;
        }
        .viewer-card {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 14px;
            padding: 0.75rem 1rem;
            background: linear-gradient(180deg, rgba(252, 250, 247, 1) 0%, rgba(248, 244, 238, 1) 100%);
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def friendly_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def render_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<NULL>"
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."
