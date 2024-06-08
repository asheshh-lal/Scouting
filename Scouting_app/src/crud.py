# crud.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models import Player
from schemas import PlayerBase

async def create_player(db: AsyncSession, player: PlayerBase):
    db_player = Player(**player.dict())
    db.add(db_player)
    await db.commit()
    await db.refresh(db_player)
    return db_player

async def get_players(db: AsyncSession, player: str = None, nation: str = None):
    query = select(Player)
    if player:
        query = query.filter(Player.player == player)
    if nation:
        query = query.filter(Player.nation == nation)
    
    result = await db.execute(query)
    return result.scalars().all()
