from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import csv
import asyncio
import aiosqlite
from aiosqlite import connect as aiosqlite_connect

DATABASE_URL = "player.db"
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Create a lock to synchronize access to the database during startup
database_lock = asyncio.Lock()

def quote_column_name(column_name):
    return f'"{column_name}"'

@app.on_event("startup")
async def upload_csv_on_startup():
    # Acquire the lock to ensure exclusive access to the database during startup
    async with database_lock:
        with open("data.csv", "r") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            columns = [quote_column_name(column) for column in reader.fieldnames]

            # Connect to the database
            async with aiosqlite_connect(DATABASE_URL) as db:
                await db.execute(f'''
                    CREATE TABLE IF NOT EXISTS data (
                        {", ".join([f"{column} TEXT" for column in columns])}
                    )
                ''')
                await db.commit()

                await db.executemany(
                    f"INSERT INTO data ({', '.join(columns)}) VALUES ({', '.join(['?' for _ in columns])})",
                    [tuple(row[column.strip('"')] for column in columns) for row in rows]
                )
                await db.commit()

@app.get("/", response_class=HTMLResponse)
async def get_players(request: Request):
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute("SELECT DISTINCT Player FROM data")
        players = await cursor.fetchall()
        player_names = [row[0] for row in players]
    return templates.TemplateResponse("index.html", {"request": request, "players": player_names})

@app.get("/players")
async def get_all_players():
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute("SELECT * FROM data")
        rows = await cursor.fetchall()
    return {"players": rows}

@app.post("/submit_player")
async def submit_player(player: str = Form(...)):
    return {"selected_player": player}
