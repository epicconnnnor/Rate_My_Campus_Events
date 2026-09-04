"""
Does the bot only say things the calendar actually supports?

Each golden question is put through the real pipeline against the frozen
events, and a second model reads the answer next to the events retrieval
actually found. The judge is not asked whether the answer is good. It is asked
one narrower thing: is every claim in it supported by the context. That is the
failure that matters, because a confident invented event is worse than no
answer at all.

Run it:

    GEMINI_API_KEY=... DATABASE_NAME=rmce_eval pytest app/test/test_hallucinations.py -v

It skips without a key rather than passing on stub vectors.
"""

import os

import pytest

from app.test.conftest import EVAL_TODAY

JUDGE_PROMPT = """\
You are checking an answer for invented information. You are not judging \
whether it is helpful, well written, or complete.

The question that was asked:
{question}

The events the system retrieved. This is the entire world of facts the answer \
was allowed to draw on:
{context}

The answer that was given:
{answer}

Decide whether every factual claim in the answer -- every event name, date, \
time, location and organizer -- is supported by the retrieved events above.

Saying that nothing matched, or offering the retrieved events as near misses, \
is supported. Rephrasing or summarising a retrieved event is supported. \
Leaving a retrieved event out is supported. Naming an event, date, time or \
place that is not in the list above is NOT supported.

Reply with one JSON object and nothing else:

  {{"supported": true or false, "reason": "one short sentence"}}
"""


def _context(matches):
    """The whole world of facts the judge is allowed to mark the answer against.

    Times go through the same formatter the answer was written from. Printing
    the stored value here instead is what made the judge fail six of nine
    questions for inventing times that were only ever a timezone apart.
    """
    from app.rag.retriever import format_event_time

    if not matches:
        return "(nothing was retrieved)"
    lines = []
    for match in matches:
        event = match.event
        lines.append(
            f"- {event.get('title')}\n"
            f"    starts: {format_event_time(event.get('starts_at'))}\n"
            f"    location: {event.get('location')}\n"
            f"    organizers: {event.get('groups')}\n"
            f"    type: {event.get('event_types')}\n"
            f"    free: {event.get('is_free')}  experience: {event.get('experience')}"
        )
    return "\n".join(lines)


def _judge(question, matches, answer_text, chat):
    from app.rag.retriever import parse_json_object

    verdict = parse_json_object(chat.complete(JUDGE_PROMPT.format(
        question=question,
        context=_context(matches),
        answer=answer_text,
    )))
    return bool(verdict.get("supported")), verdict.get("reason", "(no reason)")


# =============================================================================
# THE REPORT
# =============================================================================
#
# Every question's verdict, not only the ones that objected.
#
# This used to keep a verdict only when it was bad, which meant a green run
# said nothing at all -- no branch, no titles, no reason -- and pytest hides
# captured output for passing tests on top of that. A pass is exactly the run
# worth reading. "The judge did not object" is a weak claim for a question like
# "what isn't a lecture", and it says nothing about whether retrieval found the
# right events: an honest answer built on the wrong three is still a pass.
#
# So the titles are a column of their own. They are the difference between "the
# bot reported honestly" and "the bot reported honestly about a paddleboarding
# class it thought was comedy".

REPORT_COLUMNS = ("#", "question", "branch", "what retrieval found",
                  "verdict", "judge's reason")


def _titles(matches):
    """What retrieval actually put in front of the model."""
    return [match.event.get("title") or "(untitled)" for match in matches]


def _cell(value):
    """One table cell. A pipe or a newline in a title would close the row."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _row(number, question, answer, verdict, reason):
    titles = _titles(answer.matches)
    return "| " + " | ".join(_cell(value) for value in (
        number,
        question,
        answer.outcome,
        "<br>".join(titles) if titles else "(nothing)",
        verdict,
        reason or "",
    )) + " |"


def _write_report(heading, rows):
    """Put the table where a person will actually see it.

    $GITHUB_STEP_SUMMARY renders on the run page, which is the only place a
    passing run shows anything. Appended rather than written, so each test adds
    its own section. Falls back to stdout, where `pytest -s` gives the same
    table on a laptop.

    Nothing here may fail a run. A report that cannot be written is a report
    nobody reads, not a broken eval.
    """
    table = "\n".join([
        f"### {heading}",
        "",
        "| " + " | ".join(REPORT_COLUMNS) + " |",
        "|" + " --- |" * len(REPORT_COLUMNS),
        *rows,
        "",
    ])

    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        print("\n" + table)
        return

    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(table + "\n")
    except OSError as error:
        print(f"\n(could not write the step summary: {error})\n{table}")


def test_the_golden_set_exists(golden_questions):
    """The eval is only worth running against questions written by a person who
    has seen the real data. An empty set is a missing input, not a pass."""
    assert golden_questions, (
        "app/test/fixtures/golden_questions.json has no questions in it. "
        "They have to be written against the frozen events in "
        "app/test/fixtures/events.json -- see the README in that directory."
    )


def test_every_answer_is_grounded(live_model, eval_db, golden_questions):
    """One judged verdict per golden question, reported together.

    Deliberately not parametrized: a run that fails should show every
    ungrounded answer at once, not stop at the first.
    """
    if not golden_questions:
        pytest.skip("no golden questions written yet")

    from app.rag.answer import answer_question
    from app.rag.providers import get_judge_provider

    # Its own model: separate quota bucket, and a reader that did not write
    # the answer it is marking.
    judge = get_judge_provider()
    failures = []
    rows = []

    for number, entry in enumerate(golden_questions, start=1):
        question = entry["question"]
        answer = answer_question(question, today=EVAL_TODAY)

        expected = entry.get("expect_outcome")
        if expected and answer.outcome != expected:
            failures.append(
                f"{question!r}\n    expected outcome {expected!r}, "
                f"got {answer.outcome!r}"
            )
            # The judge is not asked. The branch is already wrong, and what it
            # thinks of an answer from the wrong branch is not worth a request.
            rows.append(_row(number, question, answer, "not judged",
                             f"expected branch {expected!r}"))
            continue

        supported, reason = _judge(question, answer.matches, answer.text, judge)
        rows.append(_row(
            number, question, answer,
            "supported" if supported else "objected",
            "" if supported else reason,
        ))
        if not supported:
            failures.append(
                f"{question!r}\n    judge: {reason}\n"
                f"    answer: {answer.text[:300]}"
            )

    _write_report("The golden questions", rows)

    assert not failures, (
        f"{len(failures)} of {len(golden_questions)} answers were not "
        "supported by what retrieval found:\n\n" + "\n\n".join(failures)
    )


def test_a_question_with_no_possible_answer_invents_nothing(live_model, eval_db):
    """Not a golden question -- a floor the bot has to clear whatever else it
    does. Nothing on campus is about this, so anything concrete is invented."""
    from app.rag.answer import answer_question
    from app.rag.providers import get_judge_provider

    question = "is there a scuba diving competition in the library basement?"
    answer = answer_question(question, today=EVAL_TODAY)
    supported, reason = _judge(
        question, answer.matches, answer.text, get_judge_provider(),
    )

    # Its own section. It is the same shape of check, and the titles matter for
    # the same reason: what a question with no possible answer drags back is
    # the clearest read on what retrieval does with a miss.
    _write_report("The floor", [_row(
        1, question, answer,
        "supported" if supported else "objected",
        "" if supported else reason,
    )])

    assert supported, f"invented something: {reason} -- {answer.text[:300]}"
