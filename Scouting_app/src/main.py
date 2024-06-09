import csv
import json
import asyncio
import aiosqlite
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from aiosqlite import connect as aiosqlite_connect
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans # Kmeans clustering

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
        with open("raw.csv", "r") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            columns = [quote_column_name(column) for column in reader.fieldnames]

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

@app.post("/submit_player", response_class=HTMLResponse)
async def submit_player(request: Request, player: str = Form(...)):
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute("SELECT Rk FROM data WHERE Player = ?", (player,))
        row = await cursor.fetchone()
    rk = row[0] if row else "No Rk found for this player"
    similar_players = await find_similar_players(rk)
    return templates.TemplateResponse("similar_players.html", {"request": request, "players": similar_players})
    
async def find_similar_players(id):
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute("SELECT * FROM data")
        rows = await cursor.fetchall()

        # Convert rows to a pandas DataFrame
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

        df_player_norm = df.copy()
        custom_mapping = {
            'GK': 1,
            'DF,FW': 4,
            'MF,FW': 8,
            'DF': 2,
            'DF,MF': 3,
            'MF,DF': 5,
            'MF': 6,
            'FW,DF': 7,
            'FW,MF': 9,
            'FW': 10
        }

        # Apply custom mapping to the 'Pos' column
        df_player_norm['Pos'] = df_player_norm['Pos'].map(custom_mapping)

        selected_features = ['Pos', 'Age','Playing Time MP', 'Performance Gls', 'Performance Ast',
            'Performance G+A', 'Performance G-PK', 'Performance Fls',
            'Performance Fld', 'Performance Crs', 'Performance Recov',
            'Expected xG', 'Expected npxG', 'Expected xAG', 'Expected xA',
            'Expected A-xAG', 'Expected G-xG', 'Expected np:G-xG',
            'Progression PrgC', 'Progression PrgP', 'Progression PrgR',
            'Tackles Tkl', 'Tackles TklW', 'Tackles Def 3rd', 'Tackles Mid 3rd',
            'Tackles Att 3rd', 'Challenges Att', 'Challenges Tkl%',
            'Challenges Lost', 'Blocks Blocks', 'Blocks Sh', 'Blocks Pass', 'Int',
            'Clr', 'Standard Sh', 'Standard SoT', 'Standard SoT%', 'Standard Sh/90',
            'Standard Dist', 'Standard FK', 'Performance GA', 'Performance SoTA',
            'Performance Saves', 'Performance Save%', 'Performance CS',
            'Performance CS%', 'Penalty Kicks PKatt', 'Penalty Kicks Save%',
            'SCA SCA', 'GCA GCA', 'Aerial Duels Won', 'Aerial Duels Lost',
            'Aerial Duels Won%', 'Total Cmp', 'Total Att', 'Total Cmp%',
            'Total TotDist', 'Total PrgDist', 'KP', '1/3', 'PPA', 'CrsPA', 'PrgP']

        scaler = MinMaxScaler()
        df_player_norm[selected_features] = scaler.fit_transform(df_player_norm[selected_features])

        num_clusters = 4
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans_model.fit(df_player_norm[selected_features])
        cluster_labels = kmeans_model.labels_

        # Add cluster labels to the player dataframe
        df_player_norm['Cluster'] = cluster_labels
        df['Cluster'] = cluster_labels

        df_player_norm['Cluster'] = df['Cluster']
        # Select the player of interest (target)
        target_player = df_player_norm[df_player_norm['Rk'] == id]
        target_features = target_player[selected_features]
        target_cluster = target_player['Cluster'].iloc[0]  # Get the cluster label of the target player

        similar_players_cluster_df = df_player_norm[df_player_norm['Cluster'] == target_cluster].copy()
        # Calculate cosine similarity between the target player and other players in the same cluster
        similarities = cosine_similarity(target_features, similar_players_cluster_df[selected_features])
        similarities = similarities[0] * 100
        # Add the 'Similarity' column to the dataframe using .loc
        similar_players_cluster_df.loc[:, 'Similarity'] = similarities

        # Sort the DataFrame based on the 'Similarity' column in descending order
        similar_players_cluster_df = similar_players_cluster_df.sort_values(by='Similarity', ascending=False)

        # Select from the second to the 11th element (remembering that indexing starts from 0)
        similar_players_cluster_df = similar_players_cluster_df.iloc[1:11]

        # Retrieve the non-normalized data from the original dataframe
        similar_players_cluster_df = df[df['Rk'].isin(similar_players_cluster_df['Rk'])]

    return similar_players_cluster_df

