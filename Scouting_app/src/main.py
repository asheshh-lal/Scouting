from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import get_db, engine
from models import Base, Player
from schemas import PlayerBase
import crud
import csv
import io
from starlette.requests import Request

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Create the database tables
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await load_data_from_csv()

async def load_data_from_csv():
    async for db in get_db():
        try:
            with open('data.csv', newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                for row in csv_reader:
                    # Handle null values or empty strings as 0
                    for key, value in row.items():
                        if value is None or value == "":
                            row[key] = 0
                    player_data = PlayerBase(**row)
                    await crud.create_player(db=db, player=player_data)
        finally:
            await db.close()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    contents = await file.read()
    csv_reader = csv.DictReader(io.StringIO(contents.decode('utf-8')))
    
    players = []
    for row in csv_reader:
        # Handle null values or empty strings as 0
        for key, value in row.items():
            if value is None or value == "":
                row[key] = 0
        player_data = PlayerBase(**row)
        players.append(await crud.create_player(db=db, player=player_data))
    
    return players

@app.get("/filter/", response_class=HTMLResponse)
async def filter_players(request: Request, player: str = None, nation: str = None, db: AsyncSession = Depends(get_db)):
    players = await crud.get_players(db, player=player, nation=nation)
    return templates.TemplateResponse("filtered.html", {"request": request, "players": players})
