from fastapi import FastAPI
import csv
import asyncio
from aiosqlite import connect as aiosqlite_connect

# SQLite database connection string
DATABASE_URL = "player.db"

# Create FastAPI app
app = FastAPI()

# Create a lock to synchronize access to the database during startup
database_lock = asyncio.Lock()

def quote_column_name(column_name):
    # Enclose the column name in double quotes to handle special characters
    return f'"{column_name}"'

@app.on_event("startup")
async def upload_csv_on_startup():
    # Acquire the lock to ensure exclusive access to the database during startup
    async with database_lock:
        # Open the CSV file and read data
        with open("data.csv", "r") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

            # Extract column names from the CSV file and quote them
            columns = [quote_column_name(column) for column in reader.fieldnames]

            # Connect to the database
            async with aiosqlite_connect(DATABASE_URL) as db:
                # Create table with dynamically determined columns
                await db.execute(f'''
                    CREATE TABLE IF NOT EXISTS data (
                        {", ".join([f"{column} TEXT" for column in columns])}
                    )
                ''')
                await db.commit()

                # Insert data into the database
                await db.executemany(
                    f"INSERT INTO data ({', '.join(columns)}) VALUES ({', '.join(['?' for _ in columns])})",
                    [tuple(row[column.strip('"')] for column in columns) for row in rows]
                )
                await db.commit()

@app.get("/")
async def home():
    return {"message": "Welcome to the CSV uploader!"}

@app.get("/players")
async def get_players():
    async with aiosqlite_connect(DATABASE_URL) as db:
        # Fetch all data from the table
        async with db.execute("SELECT * FROM data") as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]

    # Format the data as a list of dictionaries
    players = [dict(zip(columns, row)) for row in rows]

    return {"players": players}
