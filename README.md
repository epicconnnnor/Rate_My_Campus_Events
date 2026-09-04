# RateMyCampusEvents

A web application for students and organizers to view, rate, and comment on campus events. Built with FastAPI, PostgreSQL, HTMX, and SQLModel.

![Event detail](assets/event-detail.png)

## Prerequisites

Python 3.13+
Docker (for PostgreSQL)
pip (Python package manager)

## Run the chat UI locally

Five steps from a fresh clone to asking the bot a question in a browser.

**1. Start Postgres.** It must be the `pgvector` image -- semantic search keeps
its vectors in the database, and `alembic upgrade head` fails on a stock
`postgres` with `extension "vector" is not available`.

    docker run -d --name dev_pg \
      -e POSTGRES_USER=app -e POSTGRES_PASSWORD=app -e POSTGRES_DB=db \
      -p 5432:5432 pgvector/pgvector:pg16

Already have the container from a previous run? `docker start dev_pg`.

**2. Set the environment.**

    export DATABASE_HOST=localhost          # 'dev_pg' is the in-container name
    export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
    export OPENAI_API_KEY="your_key"
    export DEMO_DATE=2026-09-02

OpenAI is the default provider: `gpt-4o-mini` answers, `gpt-4.1-mini` judges,
`text-embedding-3-small` embeds at 1536 dimensions, which is the width the
`event_embeddings` column was built for. To use Gemini instead, set
`RAG_PROVIDER=gemini` and `GEMINI_API_KEY` -- the model ids follow the
provider, so there is nothing else to change.

`DEMO_DATE` is what makes this worth doing. The demo events run from
2026-09-02 to 2026-11-12, so without it the app searches a calendar the clock
has already walked past, every question comes back empty, and the bot looks
broken when the database is merely stale.

**3. One command: install, migrate, load the events, embed them, serve.**

    pip install -r requirements.txt \
      && alembic upgrade head \
      && python -m app.demo_seed --embed \
      && uvicorn app.main:app --reload

`app.demo_seed` loads the same 102 frozen events the eval judges against, and
`--embed` does the same work as `python -m app.rag.backfill`. It spends about
102 of the day's 1000 free embedding requests the first time and nothing on a
re-run, because an event whose text has not changed is skipped.

**4. Make an account.** The chat page is behind a login. Open
<http://localhost:8000/register>, register with any email and password -- it
is your local database -- and you land signed in.

**5. Ask it something.** Open <http://localhost:8000/chat> and try:

- *what's happening this weekend?*
- *any free events coming up?*
- *what's on this weekend that isn't a lecture?*
- *anything funny happening, like comedy or improv?*

The chips above the input pre-set the search rather than filtering the answer
afterwards: **Free** and **Virtual** override whatever the model read out of
your sentence, and the category chips are the calendar's own event types, read
from the database, so a chip never offers a category nothing is filed under.

### Without a Gemini key

    export RAG_PROVIDER=fake
    python -m app.demo_seed --embed

The pages, the chips, the cards and the spinner all work. The answers do not:
the fake provider returns deterministic stub vectors and the literal string
"fake completion", so it is worth exactly one thing -- seeing the interface
without spending a quota.


## HOW TO SETUP&&RUN 

### Step 1: Clone/Navigate to the Project Directory

cd "projects/ratemycampusevents"

### Step 2: Install the requirements

pip install -r requirements.txt

### Step 3: Start PostgreSQL Database

The project uses a PostgreSQL Docker container. It must have the `pgvector`
extension available -- semantic search stores its vectors in Postgres, and
`alembic upgrade head` fails on a stock `postgres` image with
`extension "vector" is not available`. The `pgvector/pgvector:pg16` image works
as a drop-in replacement.

Start it with:

sudo chgrp "$(id -gn)" /var/run/docker.sock
sudo chmod g+rw /var/run/docker.sock

./.devcontainer/tasks/svc-up.sh

**Or**

sudo docker start dev_pg

## You should see output showing the `dev_pg` container is running.

### Step 4: Database Configuration

The application uses these default PostgreSQL settings:

- **Host**: `dev_pg`
- **Port**: `5432`
- **Database**: `db`
- **Username**: `app`
- **Password**: `app`

**You can config them in `app/core/config.py`**

export DATABASE_USER="your_user"
export DATABASE_PASS="your_password"
export DATABASE_HOST="your_host"
export DATABASE_PORT="5432"
export DATABASE_NAME="your_database"

### Step 5: Run Database Migrations

The schema is managed by Alembic, not by the app. Create/update the tables with:

alembic upgrade head

Run this after every `git pull` that touches `migrations/`. Useful extras:

alembic current            # what revision the database is on
alembic upgrade head --sql # print the SQL without running it
alembic downgrade -1       # roll back one revision

### Step 5b: Sign-in with Google or GitHub (optional)

Local email/password accounts keep working with no configuration. To offer the
provider buttons as well, create an OAuth app with each provider and export its
pair:

export GOOGLE_CLIENT_ID="..."
export GOOGLE_CLIENT_SECRET="..."
export GITHUB_CLIENT_ID="..."
export GITHUB_CLIENT_SECRET="..."

The callback URL to register with the provider is
`<your host>/auth/google/callback` and `<your host>/auth/github/callback`.
A provider whose pair is missing simply does not get a button.

Accounts are linked on email, so signing in with Google as someone who already
registered with a password lands in the same account. Only addresses the
provider has verified are accepted.

Also set `SECRET_KEY` anywhere real -- the fallback is committed to this
repository and signs both the JWTs and the OAuth session cookie:

export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"

### Step 6: Build the Search Index

Embeddings need whichever provider RAG_PROVIDER names -- OpenAI by default:

export OPENAI_API_KEY="your_key"     # or RAG_PROVIDER=gemini + GEMINI_API_KEY

Changing the embedding model means the stored vectors are stale, and nothing
in the row says so: the freshness check compares the rendered text and the
source timestamp, neither of which moves when the model does. Re-embed
everything with

python -m app.rag.backfill --force

Skipping that leaves two models' vectors in one column, which does not error.
It just returns the wrong neighbours.

Then embed every event:

python -m app.ingest.localist   # pull events from the UMass calendar
python -m app.rag.backfill      # embed them

For a local run, `python -m app.demo_seed --embed` is usually what you want
instead of the first line: the frozen demo events do not need the network and
do not change week to week. See "Run the chat UI locally" above.

Both are safe to re-run. The backfill only spends an API call on events whose
text actually changed, so a second run embeds nothing.

To exercise either without a key, set `RAG_PROVIDER=fake` -- it swaps in a
deterministic local stand-in. Useful for testing, useless for real search.

### Step 7: Run the Application

Start the FastAPI server:

uvicorn app.main:app --reload

### Step 8: Access the Application

Open your web browser and navigate to:

http://localhost:8000




## Running the tests

pytest

The retrieval tests stub out the model and the database, so they need neither
an API key nor a running Postgres.

## How TO TEST IT

### 1. Create an Event

**Once logged in:**
1. Click the "Create Event" button
2. Fill in the event details:
   - Title
   - Description
   - Date & Time 
   - Location
3. Click "Create Event"
4. **The event will appear on the home page**


### 2. Delete  Comment

**On your own comment:**
1. You'll see a red "Delete" button on comments you created
2. Click "Delete"
3. Confirm the deletion
4**You should see your comment deleted**

### 3. Test Data Persistence

**Verify data persists across restarts:**
1. Create some events, comments, and ratings
2. Stop the server (Ctrl+C)
3. Restart the server: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
4. Go to `http://localhost:8000`
5. **All your data is still there!** 
