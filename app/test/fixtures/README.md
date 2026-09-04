# Evaluation fixtures

## `events.json`

102 real events pulled from the UMass calendar on 2026-09-02, frozen so that
ground truth stops moving underneath the eval. They run from **2026-09-02 to
2026-11-12**, and every eval question is asked as if today were **2026-09-02**
(`EVAL_TODAY` in `conftest.py`), so "this weekend" means one fixed weekend
forever.

The slice was taken evenly across the calendar rather than off the front, then
topped up so the rarer shapes survive:

| | count |
|---|---|
| total | 102 |
| free | 17 |
| virtual | 11 |
| hybrid | 3 |
| `Lecture/Talk/Reading` | 8 |
| no end time | 16 |
| distinct organizing groups | 19 |

Most common types: Social/Fun/Activity (36), Athletic Event (21),
Meeting/Infosession (10), Exhibit (9), Concert/Performance/Film (9),
Lecture/Talk/Reading (8).

To look through it before writing questions:

```
python -m json.tool app/test/fixtures/events.json | less
```

Refreezing it invalidates every question written against it, so don't, unless
you mean to rewrite them too.

### `.embeddings/`

Their vectors, kept between runs. Gitignored.

The events are frozen, so their vectors are too, and re-computing all 102 of
them every run was spending a tenth of the day's free embedding requests on
work that had already been done. Point `EMBEDDING_CACHE_PATH` at a directory
and the eval reuses them instead:

```
EMBEDDING_CACHE_PATH=app/test/fixtures/.embeddings \
GEMINI_API_KEY=... DATABASE_NAME=rmce_eval pytest app/test/test_hallucinations.py
```

CI sets it, and carries the directory from one run to the next.

Entries are named after a hash of the text, the model and the width, so there
is nothing to invalidate by hand: editing an event, adding a golden question or
changing `EMBEDDING_MODEL` simply misses and embeds that one thing. Deleting the
directory costs one slow run and nothing else.

## `golden_questions.json`

The questions the eval judges answers against. **These are written by hand
after reading the events above** — they are the one part of this suite that
cannot be generated, because a question invented from the schema rather than
from the data tests nothing.

```json
{
  "questions": [
    {
      "question": "what's happening this weekend that isn't a lecture?",
      "expect_outcome": "matches",
      "note": "optional, for whoever reads this later"
    }
  ]
}
```

Only `question` is required.

`expect_outcome` is optional and asserts which branch retrieval took — one of
`matches`, `alternatives`, or `empty`. Use it for the questions where the
*shape* of the response is the point: a question about a day with nothing on it
should reach `empty`, not scrape together something irrelevant.

Worth covering, given what is in the fixture:

- a plain date question ("what's on Friday")
- a date with a filter ("free things next week", "anything virtual")
- the negative case ("...that isn't a lecture") — the reason `event_types` is
  stored at all
- an organizer question ("anything from the Libraries")
- a day inside the range with nothing on it, expecting `alternatives`
- something the calendar genuinely has no answer for, expecting `empty`
