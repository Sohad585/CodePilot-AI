"""Diff helpers: a unified diff and a side-by-side HTML view.

Both views are built from the same ``difflib.SequenceMatcher`` so they always
agree. All code text is HTML-escaped before it is placed in the markup.
"""

from __future__ import annotations

import difflib
import html


def _lines(text: str) -> list[str]:
    return text.splitlines()


def unified_diff(
    original: str,
    modified: str,
    original_name: str = "original.py",
    modified_name: str = "modified.py",
) -> str:
    """Return a unified diff (empty string when nothing changed)."""
    diff = difflib.unified_diff(
        _lines(original),
        _lines(modified),
        fromfile=original_name,
        tofile=modified_name,
        lineterm="",
    )
    return "\n".join(diff)


def diff_stats(original: str, modified: str) -> tuple[int, int]:
    """Return ``(added_lines, removed_lines)``."""
    matcher = difflib.SequenceMatcher(None, _lines(original), _lines(modified))
    added = removed = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1

    return added, removed


def _cell(number: int | None, text: str, css_class: str) -> str:
    label = "" if number is None else str(number)
    body = html.escape(text, quote=False) if text else "&nbsp;"
    return (
        f'<td class="n {css_class}">{label}</td><td class="c {css_class}">{body}</td>'
    )


def _row(
    left_no: int | None,
    left_text: str,
    left_class: str,
    right_no: int | None,
    right_text: str,
    right_class: str,
) -> str:
    return (
        "<tr>"
        + _cell(left_no, left_text, left_class)
        + _cell(right_no, right_text, right_class)
        + "</tr>"
    )


def _gap_row(hidden: int) -> str:
    if hidden <= 0:
        return ""
    noun = "line" if hidden == 1 else "lines"
    return f'<tr class="cp-sep"><td colspan="4">⋯ {hidden} unchanged {noun} ⋯</td></tr>'


def side_by_side_html(original: str, modified: str, context: int | None = 3) -> str:
    """Render original and modified code side by side.

    ``context`` is the number of unchanged lines kept around each change;
    ``None`` shows both files in full.
    """
    a, b = _lines(original), _lines(modified)
    matcher = difflib.SequenceMatcher(None, a, b)

    if context is None:
        groups = [matcher.get_opcodes()]
    else:
        groups = matcher.get_grouped_opcodes(context)

    rows: list[str] = []
    previous_end = 0

    for group in groups:
        if not group:
            continue

        rows.append(_gap_row(group[0][1] - previous_end))

        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for k in range(i2 - i1):
                    rows.append(
                        _row(i1 + k + 1, a[i1 + k], "eq", j1 + k + 1, b[j1 + k], "eq")
                    )
                continue

            for k in range(max(i2 - i1, j2 - j1)):
                has_left = k < i2 - i1
                has_right = k < j2 - j1
                rows.append(
                    _row(
                        i1 + k + 1 if has_left else None,
                        a[i1 + k] if has_left else "",
                        "del" if has_left else "empty",
                        j1 + k + 1 if has_right else None,
                        b[j1 + k] if has_right else "",
                        "add" if has_right else "empty",
                    )
                )

        previous_end = group[-1][2]

    rows.append(_gap_row(len(a) - previous_end))

    return (
        '<div class="cp-diff-wrap"><table class="cp-diff">'
        "<colgroup>"
        '<col class="n"><col class="c"><col class="n"><col class="c">'
        "</colgroup>"
        '<thead><tr><th colspan="2">Original</th><th colspan="2">Modified</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )
