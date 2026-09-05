"""
Turning retrieved events into something worth reading.

Same rule as the redirect in the retriever: the model only ever sees the events
that were actually found, so there is nothing for it to invent from. Retrieval
decides what is true; this only decides how it reads.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

from app.rag.providers import ChatProvider, EmbeddingProvider, get_chat_provider
from app.rag.retriever import (FilterOverrides, Match, RetrievalResult,
                                describe_match, retrieve)

log = logging.getLogger("answer")

ANSWER_PROMPT = """\
Someone asked about events on the UMass Amherst campus. Below is every event \
you are allowed to mention -- they came from the campus calendar and they are \
the only real ones you have.

Their question: {question}

The events:
{events}

Reply like a helpful friend: one or two short sentences, no more than 45 words. \
The event cards already show names, dates, and locations, so do not repeat \
every detail. Give a quick recommendation or useful comparison instead. If \
some of the list does not fit what they asked, leave it out rather than \
stretching it. Never invent an event, a date, a location, or a detail that is \
not written above. Do not begin with filler such as "Hey!" or "There are a \
couple coming up."
"""


@dataclass
class Answer:
    # Mirrors RetrievalResult.outcome: matches / alternatives / empty
    outcome: str
    text: str
    matches: List[Match]


def build_answer_prompt(question: str, matches: List[Match]) -> str:
    return ANSWER_PROMPT.format(
        question=question,
        events="\n".join(describe_match(match) for match in matches),
    )


def phrase_answer(question: str, matches: List[Match],
                  chat: ChatProvider) -> str:
    return chat.complete(build_answer_prompt(question, matches)).strip()


def answer_question(question: str, *,
                    today: Optional[date] = None,
                    chat: Optional[ChatProvider] = None,
                    embedder: Optional[EmbeddingProvider] = None,
                    overrides: Optional[FilterOverrides] = None) -> Answer:
    """Retrieve, then say it out loud.

    The two outcomes that already carry their own wording -- a redirect to
    near misses, and the canned dead end -- are passed straight through. Only a
    real hit needs phrasing here.
    """
    chat = chat or get_chat_provider()

    result: RetrievalResult = retrieve(
        question, today=today, chat=chat, embedder=embedder,
        overrides=overrides,
    )

    if result.outcome == "matches":
        return Answer(
            outcome=result.outcome,
            text=phrase_answer(question, result.matches, chat),
            matches=result.matches,
        )

    return Answer(
        outcome=result.outcome,
        text=result.message or "",
        matches=result.matches,
    )
