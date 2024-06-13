import os
import csv
import asyncio
import aiosqlite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from aiosqlite import connect as aiosqlite_connect
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from mplsoccer import Radar, FontManager, grid

DATABASE_URL = "player.db"
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create lock to synchronize access to the database during startup
database_lock = asyncio.Lock()

def quote_column_name(column_name):
    return f'"{column_name}"'

@app.on_event("startup")
async def upload_csv_on_startup():
    # Acquire lock to ensure exclusive access to the database during startup
    async with database_lock:
        with open("Final_player_cluster_df.csv", "r") as file:
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
    async with aiosqlite_connect(DATABASE_URL) as db:
        cursor = await db.execute("SELECT DISTINCT Player FROM data")
        players = await cursor.fetchall()
        player_names = [row[0] for row in players]
    return templates.TemplateResponse("index.html", {"request": request, "players": player_names})

@app.get("/players")
async def get_all_players():
    async with aiosqlite_connect(DATABASE_URL) as db:
        cursor = await db.execute("SELECT * FROM data")
        rows = await cursor.fetchall()
    return {"players": rows}

@app.post("/submit_player", response_class=HTMLResponse)
async def submit_player(request: Request, player: str = Form(...)):
    async with aiosqlite_connect(DATABASE_URL) as db:
        cursor = await db.execute("SELECT Rk FROM data WHERE Player = ?", (player,))
        row = await cursor.fetchone()
    rk = row[0] if row else "No Rk found for this player"
    similar_players = await find_similar_players(rk)
    radar_charts = await generate_radar_charts(similar_players, rk)
    return templates.TemplateResponse("similar_players.html", {"request": request, "radar_charts": radar_charts, "players": similar_players})

async def find_similar_players(player_id):
    async with aiosqlite_connect(DATABASE_URL) as db:
        cursor = await db.execute("SELECT DISTINCT * FROM data")
        rows = await cursor.fetchall()
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

        df_player_norm['Pos'] = df_player_norm['Pos'].map(custom_mapping)

        selected_features = ['Pos', 'Age', 'Playing Time MP', 'Performance Gls', 'Performance Ast',
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

        df_player_norm['Cluster'] = df['Cluster']
        target_player = df_player_norm[df_player_norm['Rk'] == player_id]
        if target_player.empty:
            raise HTTPException(status_code=404, detail="Player not found")
        
        target_features = target_player[selected_features]
        target_cluster = target_player['Cluster'].iloc[0]  # Get the cluster label of the target player

        similar_players_cluster_df = df_player_norm[df_player_norm['Cluster'] == target_cluster].copy()
        similarities = cosine_similarity(target_features, similar_players_cluster_df[selected_features])
        similarities = similarities[0] * 100
        similar_players_cluster_df.loc[:, 'Similarity'] = similarities
        similar_players_cluster_df = similar_players_cluster_df.sort_values(by='Similarity', ascending=False)
        similar_players_cluster_df = similar_players_cluster_df.iloc[0:11]
        similar_players_cluster_df = df[df['Rk'].isin(similar_players_cluster_df['Rk'])]

    return similar_players_cluster_df


async def generate_radar_charts(similar_players_cluster_df, player_id):
    player_row = similar_players_cluster_df[similar_players_cluster_df['Rk'] == player_id]['Pos'].iloc[0]
    
    if player_row in ['FW', 'MF,FW', 'FW,MF']:
        params = [
            'Expected xG', 'Standard Sh', 'Standard SoT%',
            'Standard Sh/90', 'Aerial Duels Won%', 'Total Att',
            'Total TotDist', 'Total PrgDist'
        ]
    elif player_row in ['DF', 'DF,FW', 'DF,MF', 'FW,DF']:
        params = [
            'Expected xG', 'Tackles Tkl', 'Tackles TklW',
            'Tackles Def 3rd', 'Tackles Mid 3rd', 'Challenges Tkl%',
            'Blocks Blocks', 'Blocks Pass'
        ]
    elif player_row == 'GK':
        params = [
            "Performance GA", "Performance SoTA", "Performance Saves",
            "Performance Save%", "Performance CS", "Performance CS%",
            "Penalty Kicks PKatt", "Penalty Kicks Save%"
        ]
    elif player_row in ['MF', 'MF,DF']:
        params = [
            'Expected xA', 'Progression PrgC', 'KP', '1/3', 'PPA',
            'CrsPA', 'Total Cmp%', 'Total TotDist'
        ]
    else:
        params = []

    print(f"Parameters: {params}")

    # Convert relevant columns to numeric
    similar_players_cluster_df[params] = similar_players_cluster_df[params].apply(pd.to_numeric, errors='coerce')

    # Filter out the reference player (if applicable)
    if player_id in similar_players_cluster_df['Rk'].values:
        similar_players_cluster_df = similar_players_cluster_df[similar_players_cluster_df['Rk'] != player_id]

    # Calculate min and max values for radar chart axes
    low = []
    high = []

    static_dir = 'static'
    for filename in os.listdir(static_dir):
        if filename.endswith(".png"):
            os.remove(os.path.join(static_dir, filename))

    for param in params:
        low.append(similar_players_cluster_df[param].min())
        high.append(similar_players_cluster_df[param].max())

    # Initialize Radar chart
    radar = Radar(params, low, high,
                    round_int=[False]*len(params),
                    num_rings=4,
                    ring_width=0.4,
                    center_circle_radius=0.1
                  )

    radar_charts = []

    for idx in range(len(similar_players_cluster_df)):
        player_name = similar_players_cluster_df.iloc[idx]['Player']
        player_val = similar_players_cluster_df.iloc[idx][params].values

        fig, ax = radar.setup_axis()

        rings_inner = radar.draw_circles(ax=ax, facecolor='#ffb2b2', edgecolor='#fc5f5f')
        radar_output = radar.draw_radar(player_val, ax=ax,
                                        kwargs_radar={'facecolor': '#aa65b2'},
                                        kwargs_rings={'facecolor': '#66d8ba'})
        radar_poly, rings_outer, vertices = radar_output
        range_labels = radar.draw_range_labels(ax=ax, fontsize=15)
        param_labels = radar.draw_param_labels(ax=ax, fontsize=15)
        title = ax.set_title(f'Radar Chart for {player_name}', fontsize=20)
        title.set_position([0.5, 1.5])

        img_path = f'static/RadarChart_{player_name}.png'
        plt.savefig(img_path)
        plt.close() 
        img_path_for_template = f'RadarChart_{player_name}.png'
        radar_charts.append((player_name, img_path_for_template))

    return radar_charts






