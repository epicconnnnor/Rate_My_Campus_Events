# RateMyCampusEvents

A web application for students and organizers to view, rate, and comment on campus events. Built with FastAPI, PostgreSQL, HTMX, and SQLModel.

![Event detail](assets/event-detail.png)

## Prerequisites

Python 3.13+
Docker (for PostgreSQL)
pip (Python package manager)

## HOW TO SETUP&&RUN 

### Step 1: Clone/Navigate to the Project Directory

cd "projects/ratemycampusevents"

### Step 2: Install the requirements

pip install -r requirements.txt

### Step 3: Start PostgreSQL Database

The project uses a PostgreSQL Docker container. Start it with:

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

### Step 5: Run the Application

Start the FastAPI server:

uvicorn app.main:app --reload

### Step 6: Access the Application

Open your web browser and navigate to:

http://localhost:8000




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
