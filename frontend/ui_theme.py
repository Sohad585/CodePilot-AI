"""Dark theme styling and small HTML building blocks for the frontend.

The base dark palette comes from .streamlit/config.toml; the CSS below adds
the custom pieces (header, chips, workflow pipeline and the diff table). The
custom pieces use fixed dark colors so they look right either way.
"""

from __future__ import annotations

import html

import streamlit as st

_CSS = """
<style>
:root {
  --cp-bg: #0b0f19;
  --cp-surface: #111827;
  --cp-surface-2: #172036;
  --cp-border: #243049;
  --cp-text: #e5e7eb;
  --cp-muted: #9aa4b2;
  --cp-accent: #818cf8;
  --cp-accent-2: #22d3ee;
  --cp-ok: #34d399;
  --cp-bad: #f87171;
  --cp-warn: #fbbf24;
}

.block-container { padding-top: 2.2rem; max-width: 1400px; }

/* Header */
.cp-hero {
  padding: 1.25rem 1.5rem;
  margin-bottom: 1.25rem;
  border: 1px solid var(--cp-border);
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(99,102,241,.16), rgba(34,211,238,.08)), var(--cp-surface);
}
.cp-hero h1 {
  margin: 0; padding: 0; font-size: 2rem; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg, #a5b4fc, #22d3ee);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.cp-hero p { margin: .3rem 0 0; color: var(--cp-muted); font-size: .98rem; }

/* Section titles */
.cp-section { display: flex; align-items: center; gap: .6rem; margin: .2rem 0 .7rem; }
.cp-section .badge {
  width: 1.6rem; height: 1.6rem; border-radius: 999px; flex: none;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: .82rem; font-weight: 800; color: #0b0f19;
  background: linear-gradient(135deg, #818cf8, #22d3ee);
}
.cp-section .title { font-size: 1.1rem; font-weight: 700; color: var(--cp-text); }
.cp-section .sub { font-size: .82rem; color: var(--cp-muted); margin-left: .25rem; }

/* Chips */
.cp-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .1rem 0 .6rem; }
.cp-chip {
  display: inline-flex; align-items: center; gap: .35rem;
  padding: .12rem .65rem; border-radius: 999px;
  font-size: .78rem; font-weight: 600; line-height: 1.5;
  color: var(--cp-text); background: var(--cp-surface-2); border: 1px solid var(--cp-border);
}
.cp-chip.ok   { color: var(--cp-ok);   border-color: rgba(52,211,153,.45);  background: rgba(52,211,153,.10); }
.cp-chip.bad  { color: var(--cp-bad);  border-color: rgba(248,113,113,.45); background: rgba(248,113,113,.10); }
.cp-chip.warn { color: var(--cp-warn); border-color: rgba(251,191,36,.45);  background: rgba(251,191,36,.10); }
.cp-chip.info { color: var(--cp-accent); border-color: rgba(129,140,248,.45); background: rgba(129,140,248,.10); }

/* Workflow pipeline */
.cp-steps { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin: .4rem 0 1rem; }
.cp-step {
  display: inline-flex; align-items: center; gap: .45rem;
  padding: .4rem .85rem; border-radius: 10px; font-size: .88rem; font-weight: 600;
  background: var(--cp-surface-2); border: 1px solid var(--cp-border); color: var(--cp-muted);
}
.cp-step.done   { color: var(--cp-ok);  border-color: rgba(52,211,153,.45);  background: rgba(52,211,153,.08); }
.cp-step.failed { color: var(--cp-bad); border-color: rgba(248,113,113,.45); background: rgba(248,113,113,.08); }
.cp-arrow { color: var(--cp-muted); }

/* Diff table */
.cp-diff-wrap {
  border: 1px solid var(--cp-border); border-radius: 10px; overflow: auto;
  max-height: 640px; background: #0d1220;
}
table.cp-diff {
  width: 100%; border-collapse: collapse; table-layout: fixed;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .8rem; line-height: 1.5;
}
table.cp-diff col.n { width: 3.4rem; }
table.cp-diff th {
  position: sticky; top: 0; z-index: 1; padding: .45rem .75rem; text-align: left;
  font-family: inherit; font-weight: 700; color: var(--cp-text);
  background: var(--cp-surface-2); border-bottom: 1px solid var(--cp-border);
}
table.cp-diff td { padding: 0 .5rem; vertical-align: top; color: var(--cp-text); border: 0; }
table.cp-diff td.n {
  text-align: right; color: #6b7686; user-select: none;
  border-right: 1px solid var(--cp-border);
}
table.cp-diff td.c {
  white-space: pre-wrap; overflow-wrap: anywhere;
  border-right: 1px solid var(--cp-border);
}
table.cp-diff td.del   { background: rgba(248,113,113,.14); }
table.cp-diff td.add   { background: rgba(52,211,153,.14); }
table.cp-diff td.empty { background: rgba(255,255,255,.025); }
table.cp-diff td.n.del { color: #fca5a5; }
table.cp-diff td.n.add { color: #6ee7b7; }
table.cp-diff tr.cp-sep td {
  padding: .25rem .5rem; text-align: center; color: var(--cp-muted); font-size: .75rem;
  background: var(--cp-surface-2); border-top: 1px solid var(--cp-border); border-bottom: 1px solid var(--cp-border);
}
</style>
"""


def inject_theme() -> None:
    """Inject the custom CSS once per run."""
    st.markdown(_CSS, unsafe_allow_html=True)


def chip(label: str, tone: str = "neutral") -> str:
    """Return the HTML of a small status chip. ``tone``: ok, bad, warn, info."""
    css = tone if tone in {"ok", "bad", "warn", "info"} else ""
    return f'<span class="cp-chip {css}">{html.escape(label)}</span>'


def chips(*items: str) -> None:
    """Render several chips (already built with :func:`chip`) in one row."""
    st.html(f'<div class="cp-chips">{"".join(items)}</div>')


def hero() -> None:
    st.html(
        '<div class="cp-hero">'
        "<h1>CodePilot AI</h1>"
        "<p>Autonomous Python debugging &amp; refactoring. Upload a file, "
        "describe the task, and review every change before you take it.</p>"
        "</div>"
    )


def section(step: int, title: str, subtitle: str = "") -> None:
    sub = f'<span class="sub">{html.escape(subtitle)}</span>' if subtitle else ""
    st.html(
        '<div class="cp-section">'
        f'<span class="badge">{step}</span>'
        f'<span class="title">{html.escape(title)}</span>{sub}'
        "</div>"
    )
