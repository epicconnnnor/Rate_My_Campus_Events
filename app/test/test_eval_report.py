"""
The eval's report, checked without paying for an eval.

The table itself is pure formatting over data the judged run already has, so
it is worth testing where a key and a database are not: these run in the unit
job, on every push, while the run they describe is nightly.

A silently malformed table is the failure worth guarding. One pipe in an event
title closes the row early, and the column that goes missing is the one nobody
notices is missing -- so the escaping gets a test rather than a reading.
"""

from app.test.test_hallucinations import (REPORT_COLUMNS, _cell, _row,
                                          _titles, _write_report)


class FakeMatch:
    def __init__(self, title):
        self.event = {"title": title}


class FakeAnswer:
    def __init__(self, outcome, titles):
        self.outcome = outcome
        self.matches = [FakeMatch(title) for title in titles]


def cells(row):
    """The row back as fields, the way a markdown renderer reads it."""
    assert row.startswith("| ") and row.endswith(" |")
    # An escaped pipe is content, not a separator.
    return [field.strip().replace("\\|", "|")
            for field in row[2:-2].split(" | ")]


# =============================================================================
# CELLS
# =============================================================================


def test_a_pipe_in_a_title_does_not_close_the_row():
    """'Art | Reception' is an ordinary title and a broken table."""
    row = _row(1, "q", FakeAnswer("matches", ["Art | Reception"]), "supported", "")
    assert len(cells(row)) == len(REPORT_COLUMNS)
    assert "Art | Reception" in cells(row)[3]


def test_a_newline_in_a_reason_stays_on_one_line():
    """The judge writes a sentence; nothing says it is one line."""
    row = _row(1, "q", FakeAnswer("empty", []), "objected", "two\nlines")
    assert "\n" not in row
    assert cells(row)[5] == "two lines"


def test_a_cell_is_stripped_and_stringified():
    assert _cell("  padded  ") == "padded"
    assert _cell(7) == "7"


# =============================================================================
# ROWS
# =============================================================================


def test_a_row_carries_the_branch_and_the_titles():
    """The two columns a passing run exists to show: an honest answer over the
    wrong events reads as a pass everywhere else."""
    row = cells(_row(
        8, "anything funny happening, like comedy or improv?",
        FakeAnswer("alternatives", ["Stand-Up Paddleboarding Basics"]),
        "supported", "",
    ))
    assert row[0] == "8"
    assert row[2] == "alternatives"
    assert row[3] == "Stand-Up Paddleboarding Basics"
    assert row[4] == "supported"


def test_several_titles_are_listed_within_the_one_cell():
    row = cells(_row(1, "q", FakeAnswer("matches", ["One", "Two"]), "supported", ""))
    assert row[3] == "One<br>Two"


def test_retrieving_nothing_says_so_rather_than_leaving_a_blank():
    """An empty cell reads as a bug in the report. This is a real result."""
    row = cells(_row(1, "q", FakeAnswer("empty", []), "supported", ""))
    assert row[3] == "(nothing)"


def test_an_untitled_event_does_not_produce_the_word_none():
    assert _titles([FakeMatch(None)]) == ["(untitled)"]


# =============================================================================
# WRITING IT OUT
# =============================================================================


def test_the_report_appends_so_each_test_keeps_its_section(tmp_path,
                                                           monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    _write_report("The golden questions", [
        _row(1, "q", FakeAnswer("matches", ["One"]), "supported", ""),
    ])
    _write_report("The floor", [
        _row(1, "scuba?", FakeAnswer("empty", []), "supported", ""),
    ])

    written = summary.read_text(encoding="utf-8")
    assert "### The golden questions" in written
    assert "### The floor" in written
    assert written.count("| # | question |") == 2


def test_no_summary_file_falls_back_to_stdout(monkeypatch, capsys):
    """A laptop has no $GITHUB_STEP_SUMMARY. `pytest -s` should still show it."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    _write_report("Heading", [_row(1, "q", FakeAnswer("empty", []), "supported", "")])
    assert "### Heading" in capsys.readouterr().out


def test_a_report_that_cannot_be_written_does_not_fail_the_run(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    """Fifteen minutes and a slice of a daily quota are not worth losing to a
    file that would not open."""
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "nope" / "s.md"))
    _write_report("Heading", [_row(1, "q", FakeAnswer("empty", []), "supported", "")])
    assert "could not write the step summary" in capsys.readouterr().out


def test_the_header_matches_the_number_of_fields_in_a_row():
    """The two are written apart, so a column added to one and not the other
    misaligns every row under it."""
    row = _row(1, "q", FakeAnswer("empty", []), "supported", "reason")
    assert len(cells(row)) == len(REPORT_COLUMNS)
