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
    if not matches:
        return "(nothing was retrieved)"
    lines = []
    for match in matches:
        event = match.event
        lines.append(
            f"- {event.get('title')}\n"
            f"    starts: {event.get('starts_at')}\n"
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

    for entry in golden_questions:
        question = entry["question"]
        answer = answer_question(question, today=EVAL_TODAY)

        expected = entry.get("expect_outcome")
        if expected and answer.outcome != expected:
            failures.append(
                f"{question!r}\n    expected outcome {expected!r}, "
                f"got {answer.outcome!r}"
            )
            continue

        supported, reason = _judge(question, answer.matches, answer.text, judge)
        if not supported:
            failures.append(
                f"{question!r}\n    judge: {reason}\n"
                f"    answer: {answer.text[:300]}"
            )

    assert not failures, (
        f"{len(failures)} of {len(golden_questions)} answers were not "
        "supported by what retrieval found:\n\n" + "\n\n".join(failures)
    )


def test_a_question_with_no_possible_answer_invents_nothing(live_model, eval_db):
    """Not a golden question -- a floor the bot has to clear whatever else it
    does. Nothing on campus is about this, so anything concrete is invented."""
    from app.rag.answer import answer_question
    from app.rag.providers import get_judge_provider

    answer = answer_question(
        "is there a scuba diving competition in the library basement?",
        today=EVAL_TODAY,
    )
    supported, reason = _judge(
        "is there a scuba diving competition in the library basement?",
        answer.matches, answer.text, get_judge_provider(),
    )
    assert supported, f"invented something: {reason} -- {answer.text[:300]}"
