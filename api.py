import string
import random
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    select,
    event,
    text,
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, relationship

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

DATABASE_URL = "sqlite+aiosqlite:///poker.db"

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


# Enable SQLite foreign key enforcement
@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Base(DeclarativeBase):
    pass


class RoomModel(Base):
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    owner_username = Column(String, nullable=False)
    buy_in = Column(Float, nullable=False)
    small_blind = Column(Float, nullable=False, default=0.25)
    big_blind = Column(Float, nullable=False, default=0.50)
    dealer_username = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    players = relationship("PlayerModel", back_populates="room", lazy="selectin")
    transfers = relationship("TransferModel", back_populates="room", lazy="selectin")


class PlayerModel(Base):
    __tablename__ = "players"
    __table_args__ = (UniqueConstraint("room_id", "username", name="uq_room_player"),)

    id = Column(Integer, primary_key=True)
    room_id = Column(Integer, ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False)
    username = Column(String, nullable=False)
    initial_balance = Column(Float, nullable=False)
    current_balance = Column(Float, nullable=False)
    position = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, default=True, nullable=False)
    sit_out = Column(Boolean, default=False, nullable=False)
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    room = relationship("RoomModel", back_populates="players")


class TransferModel(Base):
    __tablename__ = "transfers"

    id = Column(Integer, primary_key=True)
    room_id = Column(Integer, ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False)
    from_username = Column(String, nullable=False)
    to_username = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, default="pending", nullable=False)  # pending | approved | rejected
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    resolved_at = Column(DateTime, nullable=True)

    room = relationship("RoomModel", back_populates="transfers")


ROUND_ORDER = ["preflop", "flop", "turn", "river"]
ROUND_PROMPTS = {
    "preflop": "Pre-flop betting complete. Deal the flop (3 community cards).",
    "flop": "Flop betting complete. Deal the turn (1 community card).",
    "turn": "Turn betting complete. Deal the river (1 community card).",
    "river": "River betting complete. Showdown — determine the winner.",
}


class HandModel(Base):
    __tablename__ = "hands"

    id = Column(Integer, primary_key=True)
    room_id = Column(Integer, ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, default="active", nullable=False)  # active | round_complete | complete | settled
    current_round = Column(String, default="preflop", nullable=False)
    current_bet = Column(Float, default=0, nullable=False)
    action_on_username = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    hand_players = relationship("HandPlayerModel", back_populates="hand", lazy="selectin")


class HandPlayerModel(Base):
    __tablename__ = "hand_players"
    __table_args__ = (UniqueConstraint("hand_id", "username", name="uq_hand_player"),)

    id = Column(Integer, primary_key=True)
    hand_id = Column(Integer, ForeignKey("hands.id", ondelete="CASCADE"), nullable=False)
    username = Column(String, nullable=False)
    status = Column(String, default="active", nullable=False)  # active | folded | all_in
    bet_this_round = Column(Float, default=0, nullable=False)
    total_bet = Column(Float, default=0, nullable=False)
    has_acted = Column(Boolean, default=False, nullable=False)

    hand = relationship("HandModel", back_populates="hand_players")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CreateRoomRequest(BaseModel):
    name: str
    owner_username: str
    buy_in: float = Field(gt=0)


class RoomResponse(BaseModel):
    code: str
    name: str
    owner_username: str
    buy_in: float
    small_blind: float
    big_blind: float
    dealer_username: Optional[str] = None
    created_at: datetime


class PlayerResponse(BaseModel):
    username: str
    initial_balance: float
    current_balance: float
    position: int
    is_active: bool


class RoomDetailResponse(RoomResponse):
    players: list[PlayerResponse]


class JoinRequest(BaseModel):
    username: str


class LeaveRequest(BaseModel):
    username: str


class TransferRequest(BaseModel):
    from_username: str
    to_username: str
    amount: float = Field(gt=0)


class TransferResponse(BaseModel):
    id: int
    from_username: str
    to_username: str
    amount: float
    status: str
    created_at: datetime
    resolved_at: Optional[datetime] = None


class OwnerActionRequest(BaseModel):
    owner_username: str


class PotInfo(BaseModel):
    amount: float
    eligible: list[str]


class HandPlayerInfo(BaseModel):
    username: str
    status: str
    bet_this_round: float
    total_bet: float


class HandResponse(BaseModel):
    id: int
    status: str
    current_round: str
    current_bet: float
    action_on: Optional[str] = None
    pots: list[PotInfo]
    players: list[HandPlayerInfo]
    dealer_prompt: Optional[str] = None
    owner_prompt: Optional[str] = None


class ActionRequest(BaseModel):
    username: str
    action: str  # check | call | raise | fold
    amount: Optional[float] = None  # total bet for the round when raising


class SettlePotWinner(BaseModel):
    pot_index: int
    winners: list[str]


class SettleRequest(BaseModel):
    owner_username: str
    pot_winners: list[SettlePotWinner]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Poker Manager")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Migrate: add blind columns if missing
        try:
            await conn.execute(text("ALTER TABLE rooms ADD COLUMN small_blind FLOAT NOT NULL DEFAULT 0.25"))
        except Exception:
            pass
        try:
            await conn.execute(text("ALTER TABLE rooms ADD COLUMN big_blind FLOAT NOT NULL DEFAULT 0.50"))
        except Exception:
            pass
        try:
            await conn.execute(text("ALTER TABLE players ADD COLUMN sit_out BOOLEAN NOT NULL DEFAULT 0"))
        except Exception:
            pass


def _generate_code(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


async def _get_room(session: AsyncSession, code: str) -> RoomModel:
    result = await session.execute(select(RoomModel).where(RoomModel.code == code))
    room = result.scalar_one_or_none()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


def _compute_pots(hand_players: list) -> list[PotInfo]:
    """Compute main pot and side pots from player total bets."""
    all_with_bets = [hp for hp in hand_players if hp.total_bet > 0]
    non_folded = [hp for hp in hand_players if hp.status != "folded"]

    if not all_with_bets:
        return []

    levels = sorted(set(hp.total_bet for hp in all_with_bets))
    pots: list[PotInfo] = []
    prev = 0.0

    for level in levels:
        contributors = sum(1 for hp in all_with_bets if hp.total_bet > prev)
        amount = (level - prev) * contributors
        eligible = sorted(hp.username for hp in non_folded if hp.total_bet >= level)

        if amount > 0 and eligible:
            # Merge with previous pot if same eligible set
            if pots and pots[-1].eligible == eligible:
                pots[-1].amount += amount
            else:
                pots.append(PotInfo(amount=amount, eligible=eligible))

        prev = level

    return pots


def _find_next_actor(
    hand_players: list,
    room_players: list,
    after_username: Optional[str],
    dealer_username: str,
) -> Optional[str]:
    """Find the next active player who hasn't acted, in position order."""
    position_map = {p.username: p.position for p in room_players}
    candidates = [hp for hp in hand_players if hp.status == "active" and not hp.has_acted]

    if not candidates:
        return None

    candidates.sort(key=lambda hp: position_map.get(hp.username, 0))

    if after_username is None:
        # Start of round — first player after dealer
        dealer_pos = position_map.get(dealer_username, -1)
        after = [hp for hp in candidates if position_map.get(hp.username, 0) > dealer_pos]
        return after[0].username if after else candidates[0].username

    current_pos = position_map.get(after_username, 0)
    after = [hp for hp in candidates if position_map.get(hp.username, 0) > current_pos]
    return after[0].username if after else candidates[0].username


def _build_hand_response(hand: HandModel, dealer_prompt: Optional[str] = None) -> HandResponse:
    pots = _compute_pots(hand.hand_players)

    # Build owner prompt when the hand needs settling
    owner_prompt = None
    if hand.status == "complete" and pots:
        pot_lines = []
        for i, pot in enumerate(pots):
            pot_lines.append(f"  Pot {i}: ${pot.amount:.2f} — eligible: {', '.join(pot.eligible)}")
        pot_summary = "\n".join(pot_lines)
        owner_prompt = "Hand complete. Assign payouts to the winning hand(s)."

    return HandResponse(
        id=hand.id,
        status=hand.status,
        current_round=hand.current_round,
        current_bet=hand.current_bet,
        action_on=hand.action_on_username,
        pots=pots,
        players=[
            HandPlayerInfo(
                username=hp.username,
                status=hp.status,
                bet_this_round=hp.bet_this_round,
                total_bet=hp.total_bet,
            )
            for hp in hand.hand_players
        ],
        dealer_prompt=dealer_prompt,
        owner_prompt=owner_prompt,
    )


# ---------------------------------------------------------------------------
# Room endpoints
# ---------------------------------------------------------------------------


@app.post("/rooms", response_model=RoomResponse, status_code=201)
async def create_room(req: CreateRoomRequest):
    async with SessionLocal() as session:
        code = _generate_code()
        # Ensure uniqueness
        while (await session.execute(select(RoomModel).where(RoomModel.code == code))).scalar_one_or_none():
            code = _generate_code()

        room = RoomModel(
            code=code,
            name=req.name,
            owner_username=req.owner_username,
            buy_in=req.buy_in,
        )
        session.add(room)
        await session.flush()

        # Auto-join the owner
        owner_player = PlayerModel(
            room_id=room.id,
            username=req.owner_username,
            initial_balance=req.buy_in,
            current_balance=req.buy_in,
            position=0,
        )
        session.add(owner_player)
        await session.commit()
        await session.refresh(room)

        return RoomResponse(
            code=room.code,
            name=room.name,
            owner_username=room.owner_username,
            buy_in=room.buy_in,
            small_blind=room.small_blind,
            big_blind=room.big_blind,
            dealer_username=room.dealer_username,
            created_at=room.created_at,
        )


@app.get("/rooms/{code}", response_model=RoomDetailResponse)
async def get_room(code: str):
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        players = [
            PlayerResponse(
                username=p.username,
                initial_balance=p.initial_balance,
                current_balance=p.current_balance,
                position=p.position,
                is_active=p.is_active,
            )
            for p in sorted(room.players, key=lambda p: p.position)
        ]
        return RoomDetailResponse(
            code=room.code,
            name=room.name,
            owner_username=room.owner_username,
            buy_in=room.buy_in,
            small_blind=room.small_blind,
            big_blind=room.big_blind,
            dealer_username=room.dealer_username,
            created_at=room.created_at,
            players=players,
        )


@app.delete("/rooms/{code}", status_code=204)
async def delete_room(code: str, req: OwnerActionRequest):
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        if room.owner_username != req.owner_username:
            raise HTTPException(status_code=403, detail="Only the room owner can delete the room")
        await session.delete(room)
        await session.commit()


# ---------------------------------------------------------------------------
# Player endpoints
# ---------------------------------------------------------------------------


@app.post("/rooms/{code}/join", response_model=PlayerResponse)
async def join_room(code: str, req: JoinRequest):
    async with SessionLocal() as session:
        room = await _get_room(session, code)

        # Check if a hand is in progress
        active_hand = await session.execute(
            select(HandModel).where(
                HandModel.room_id == room.id,
                HandModel.status.in_(["active", "round_complete"]),
            )
        )
        hand_in_progress = active_hand.scalar_one_or_none() is not None

        # Check if player already exists in room
        result = await session.execute(
            select(PlayerModel).where(
                PlayerModel.room_id == room.id,
                PlayerModel.username == req.username,
            )
        )
        player = result.scalar_one_or_none()

        if player:
            # Rejoin — reactivate with old balance
            player.is_active = True
            if hand_in_progress:
                player.sit_out = True
            await session.commit()
            await session.refresh(player)
        else:
            # Assign next position
            max_pos_result = await session.execute(
                select(PlayerModel.position)
                .where(PlayerModel.room_id == room.id)
                .order_by(PlayerModel.position.desc())
                .limit(1)
            )
            max_pos = max_pos_result.scalar_one_or_none()
            next_pos = (max_pos + 1) if max_pos is not None else 0

            player = PlayerModel(
                room_id=room.id,
                username=req.username,
                initial_balance=room.buy_in,
                current_balance=room.buy_in,
                position=next_pos,
                sit_out=hand_in_progress,
            )
            session.add(player)
            await session.commit()
            await session.refresh(player)

        return PlayerResponse(
            username=player.username,
            initial_balance=player.initial_balance,
            current_balance=player.current_balance,
            position=player.position,
            is_active=player.is_active,
        )


@app.post("/rooms/{code}/leave", response_model=PlayerResponse)
async def leave_room(code: str, req: LeaveRequest):
    async with SessionLocal() as session:
        room = await _get_room(session, code)

        result = await session.execute(
            select(PlayerModel).where(
                PlayerModel.room_id == room.id,
                PlayerModel.username == req.username,
            )
        )
        player = result.scalar_one_or_none()
        if not player:
            raise HTTPException(status_code=404, detail="Player not found in this room")

        if not player.is_active:
            raise HTTPException(status_code=400, detail="Player has already left")

        player.is_active = False
        await session.commit()
        await session.refresh(player)

        return PlayerResponse(
            username=player.username,
            initial_balance=player.initial_balance,
            current_balance=player.current_balance,
            position=player.position,
            is_active=player.is_active,
        )


# ---------------------------------------------------------------------------
# Kick endpoint
# ---------------------------------------------------------------------------


class KickRequest(BaseModel):
    owner_username: str
    username: str


@app.post("/rooms/{code}/kick", response_model=PlayerResponse)
async def kick_player(code: str, req: KickRequest):
    """Owner removes a player from the active table."""
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        if room.owner_username != req.owner_username:
            raise HTTPException(status_code=403, detail="Only the room owner can kick players")

        if req.username == room.owner_username:
            raise HTTPException(status_code=400, detail="Cannot kick the room owner")

        result = await session.execute(
            select(PlayerModel).where(
                PlayerModel.room_id == room.id,
                PlayerModel.username == req.username,
            )
        )
        player = result.scalar_one_or_none()
        if not player:
            raise HTTPException(status_code=404, detail="Player not found in this room")

        player.is_active = False
        await session.commit()
        await session.refresh(player)

        return PlayerResponse(
            username=player.username,
            initial_balance=player.initial_balance,
            current_balance=player.current_balance,
            position=player.position,
            is_active=player.is_active,
        )


# ---------------------------------------------------------------------------
# Buy-in adjustment endpoint
# ---------------------------------------------------------------------------


class AdjustBuyInRequest(BaseModel):
    owner_username: str
    username: str
    amount: float  # positive = add chips, negative = remove chips
    adjust_buy_in: bool = True  # if False, only changes current_balance (not buy-in)


@app.post("/rooms/{code}/buy-in", response_model=PlayerResponse)
async def adjust_buy_in(code: str, req: AdjustBuyInRequest):
    """Owner adds or removes chips from a player's stack.

    If adjust_buy_in is True (default): changes both initial_balance and current_balance.
    If adjust_buy_in is False: only changes current_balance (e.g. corrections).
    """
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        if room.owner_username != req.owner_username:
            raise HTTPException(status_code=403, detail="Only the room owner can adjust buy-ins")

        result = await session.execute(
            select(PlayerModel).where(
                PlayerModel.room_id == room.id,
                PlayerModel.username == req.username,
            )
        )
        player = result.scalar_one_or_none()
        if not player:
            raise HTTPException(status_code=404, detail="Player not found in this room")

        if req.amount < 0 and player.current_balance + req.amount < 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot remove {abs(req.amount)} — player only has {player.current_balance} in current balance",
            )

        if req.adjust_buy_in:
            player.initial_balance += req.amount
        player.current_balance += req.amount

        await session.commit()
        await session.refresh(player)

        return PlayerResponse(
            username=player.username,
            initial_balance=player.initial_balance,
            current_balance=player.current_balance,
            position=player.position,
            is_active=player.is_active,
        )


# ---------------------------------------------------------------------------
# Reorder endpoint
# ---------------------------------------------------------------------------


class ReorderRequest(BaseModel):
    owner_username: str
    order: list[str]  # list of usernames in desired order


@app.put("/rooms/{code}/reorder", response_model=list[PlayerResponse])
async def reorder_players(code: str, req: ReorderRequest):
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        if room.owner_username != req.owner_username:
            raise HTTPException(status_code=403, detail="Only the room owner can reorder players")

        # Build a map of current players
        result = await session.execute(
            select(PlayerModel).where(PlayerModel.room_id == room.id)
        )
        players_by_name = {p.username: p for p in result.scalars().all()}

        # Validate all usernames in the order list exist in the room
        for uname in req.order:
            if uname not in players_by_name:
                raise HTTPException(status_code=400, detail=f"Player '{uname}' is not in this room")

        # Assign positions based on provided order
        for i, uname in enumerate(req.order):
            players_by_name[uname].position = i

        # Players not in the order list get positions after the listed ones
        unlisted = [p for name, p in players_by_name.items() if name not in req.order]
        for j, p in enumerate(sorted(unlisted, key=lambda x: x.position)):
            p.position = len(req.order) + j

        await session.commit()

        # Return all players sorted by new position
        all_players = sorted(players_by_name.values(), key=lambda p: p.position)
        return [
            PlayerResponse(
                username=p.username,
                initial_balance=p.initial_balance,
                current_balance=p.current_balance,
                position=p.position,
                is_active=p.is_active,
            )
            for p in all_players
        ]


# ---------------------------------------------------------------------------
# Dealer endpoints
# ---------------------------------------------------------------------------


class SetBlindsRequest(BaseModel):
    owner_username: str
    small_blind: float = Field(gt=0)
    big_blind: float = Field(gt=0)


@app.put("/rooms/{code}/blinds", response_model=RoomResponse)
async def set_blinds(code: str, req: SetBlindsRequest):
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        if room.owner_username != req.owner_username:
            raise HTTPException(status_code=403, detail="Only the room owner can set blinds")
        room.small_blind = req.small_blind
        room.big_blind = req.big_blind
        await session.commit()
        await session.refresh(room)
        return RoomResponse(
            code=room.code,
            name=room.name,
            owner_username=room.owner_username,
            buy_in=room.buy_in,
            small_blind=room.small_blind,
            big_blind=room.big_blind,
            dealer_username=room.dealer_username,
            created_at=room.created_at,
        )


class SetDealerRequest(BaseModel):
    owner_username: str
    dealer_username: str


class DealerResponse(BaseModel):
    dealer_username: str


@app.put("/rooms/{code}/dealer", response_model=DealerResponse)
async def set_dealer(code: str, req: SetDealerRequest):
    """Owner picks the initial dealer."""
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        if room.owner_username != req.owner_username:
            raise HTTPException(status_code=403, detail="Only the room owner can set the dealer")

        # Verify the chosen dealer is an active player
        result = await session.execute(
            select(PlayerModel).where(
                PlayerModel.room_id == room.id,
                PlayerModel.username == req.dealer_username,
                PlayerModel.is_active == True,  # noqa: E712
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail=f"Player '{req.dealer_username}' is not active in this room")

        room.dealer_username = req.dealer_username
        await session.commit()

        return DealerResponse(dealer_username=room.dealer_username)


@app.post("/rooms/{code}/dealer/next", response_model=DealerResponse)
async def next_dealer(code: str):
    """Advance the dealer to the next active player in position order."""
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        if not room.dealer_username:
            raise HTTPException(status_code=400, detail="No dealer has been set yet. Use PUT /rooms/{code}/dealer first.")

        # Get active players sorted by position
        result = await session.execute(
            select(PlayerModel).where(
                PlayerModel.room_id == room.id,
                PlayerModel.is_active == True,  # noqa: E712
            ).order_by(PlayerModel.position)
        )
        active_players = result.scalars().all()

        if len(active_players) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 active players to rotate dealer")

        # Find current dealer's index and pick the next one (wrapping around)
        usernames = [p.username for p in active_players]
        try:
            current_idx = usernames.index(room.dealer_username)
        except ValueError:
            # Current dealer left the room — start from position 0
            current_idx = -1

        next_idx = (current_idx + 1) % len(usernames)
        room.dealer_username = usernames[next_idx]
        await session.commit()

        return DealerResponse(dealer_username=room.dealer_username)


# ---------------------------------------------------------------------------
# Hand endpoints
# ---------------------------------------------------------------------------


@app.post("/rooms/{code}/hands", response_model=HandResponse, status_code=201)
async def start_hand(code: str):
    """Start a new hand. Requires a dealer to be set and at least 2 active players."""
    async with SessionLocal() as session:
        room = await _get_room(session, code)

        if not room.dealer_username:
            raise HTTPException(status_code=400, detail="Set a dealer before starting a hand")

        # No hand already in progress
        existing = await session.execute(
            select(HandModel).where(
                HandModel.room_id == room.id,
                HandModel.status.in_(["active", "round_complete"]),
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="A hand is already in progress")

        active_room_players = [p for p in room.players if p.is_active]
        if len(active_room_players) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 active players")

        hand = HandModel(room_id=room.id)
        session.add(hand)
        await session.flush()

        for p in active_room_players:
            hp = HandPlayerModel(hand_id=hand.id, username=p.username)
            if p.sit_out:
                hp.status = "folded"
                hp.has_acted = True
            session.add(hp)
            p.sit_out = False

        await session.flush()
        await session.refresh(hand)

        # --- Post blinds ---
        SMALL_BLIND = room.small_blind
        BIG_BLIND = room.big_blind

        # Sort active players by position, find dealer position
        sorted_players = sorted(active_room_players, key=lambda p: p.position)
        dealer_pos = next((p.position for p in sorted_players if p.username == room.dealer_username), -1)

        # Get players after dealer (wrapping around)
        after_dealer = [p for p in sorted_players if p.position > dealer_pos]
        before_dealer = [p for p in sorted_players if p.position <= dealer_pos]
        ordered = after_dealer + before_dealer

        sb_player = ordered[0]  # first after dealer = small blind
        bb_player = ordered[1]  # second after dealer = big blind

        # Post small blind
        sb_hp = next(hp for hp in hand.hand_players if hp.username == sb_player.username)
        sb_amount = min(SMALL_BLIND, sb_player.current_balance)
        sb_player.current_balance -= sb_amount
        sb_hp.bet_this_round = sb_amount
        sb_hp.total_bet = sb_amount

        # Post big blind
        bb_hp = next(hp for hp in hand.hand_players if hp.username == bb_player.username)
        bb_amount = min(BIG_BLIND, bb_player.current_balance)
        bb_player.current_balance -= bb_amount
        bb_hp.bet_this_round = bb_amount
        bb_hp.total_bet = bb_amount

        hand.current_bet = BIG_BLIND

        # First to act preflop is the player after the big blind
        first = _find_next_actor(hand.hand_players, room.players, bb_player.username, room.dealer_username)
        hand.action_on_username = first

        await session.commit()
        await session.refresh(hand)
        return _build_hand_response(hand)


@app.get("/rooms/{code}/hands/current", response_model=HandResponse)
async def get_current_hand(code: str):
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        result = await session.execute(
            select(HandModel).where(
                HandModel.room_id == room.id,
                HandModel.status.in_(["active", "round_complete", "complete"]),
            ).order_by(HandModel.created_at.desc()).limit(1)
        )
        hand = result.scalar_one_or_none()
        if not hand:
            raise HTTPException(status_code=404, detail="No active hand")

        prompt = ROUND_PROMPTS.get(hand.current_round) if hand.status == "round_complete" else None
        return _build_hand_response(hand, prompt)


@app.post("/rooms/{code}/hands/current/action", response_model=HandResponse)
async def take_action(code: str, req: ActionRequest):
    async with SessionLocal() as session:
        room = await _get_room(session, code)

        result = await session.execute(
            select(HandModel).where(
                HandModel.room_id == room.id,
                HandModel.status == "active",
            )
        )
        hand = result.scalar_one_or_none()
        if not hand:
            raise HTTPException(status_code=400, detail="No active hand (or round is complete — advance to next round)")

        if hand.action_on_username != req.username:
            raise HTTPException(
                status_code=400,
                detail=f"It's {hand.action_on_username}'s turn, not {req.username}'s",
            )

        if req.action not in ("check", "call", "raise", "fold"):
            raise HTTPException(status_code=400, detail="Action must be check, call, raise, or fold")

        hp = next((p for p in hand.hand_players if p.username == req.username), None)
        rp = next((p for p in room.players if p.username == req.username), None)
        if not hp or not rp:
            raise HTTPException(status_code=400, detail="Player not found")

        # ---- Process action ----
        if req.action == "fold":
            hp.status = "folded"
            hp.has_acted = True

        elif req.action == "check":
            if hp.bet_this_round < hand.current_bet:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot check — current bet is {hand.current_bet}, you've bet {hp.bet_this_round}. Call or raise.",
                )
            hp.has_acted = True

        elif req.action == "call":
            to_call = hand.current_bet - hp.bet_this_round
            if to_call <= 0:
                raise HTTPException(status_code=400, detail="Nothing to call — use check")

            if rp.current_balance <= to_call:
                # All-in (can't cover the full call)
                added = rp.current_balance
                rp.current_balance = 0
                hp.bet_this_round += added
                hp.total_bet += added
                hp.status = "all_in"
            else:
                rp.current_balance -= to_call
                hp.bet_this_round = hand.current_bet
                hp.total_bet += to_call
            hp.has_acted = True

        elif req.action == "raise":
            if req.amount is None or req.amount <= hand.current_bet:
                raise HTTPException(
                    status_code=400,
                    detail=f"Raise amount must exceed current bet ({hand.current_bet})",
                )
            additional = req.amount - hp.bet_this_round

            if rp.current_balance <= additional:
                # All-in raise
                added = rp.current_balance
                rp.current_balance = 0
                hp.bet_this_round += added
                hp.total_bet += added
                hp.status = "all_in"
                if hp.bet_this_round > hand.current_bet:
                    hand.current_bet = hp.bet_this_round
                    for other in hand.hand_players:
                        if other.username != hp.username and other.status == "active":
                            other.has_acted = False
            else:
                rp.current_balance -= additional
                hp.bet_this_round = req.amount
                hp.total_bet += additional
                hand.current_bet = req.amount
                # Re-open action to all other active players
                for other in hand.hand_players:
                    if other.username != hp.username and other.status == "active":
                        other.has_acted = False
            hp.has_acted = True

        # ---- Post-action checks ----
        dealer_prompt = None

        # Everyone folded except one?
        non_folded = [p for p in hand.hand_players if p.status != "folded"]
        if len(non_folded) == 1:
            winner = non_folded[0]
            winner_rp = next(p for p in room.players if p.username == winner.username)
            total_pot = sum(p.total_bet for p in hand.hand_players)
            winner_rp.current_balance += total_pot
            hand.status = "complete"
            hand.action_on_username = None

            await session.commit()
            await session.refresh(hand)
            return _build_hand_response(hand, f"{winner.username} wins — all others folded.")

        # Is the betting round complete?
        active = [p for p in hand.hand_players if p.status == "active"]
        round_done = (
            all(p.has_acted and p.bet_this_round == hand.current_bet for p in active)
            if active
            else True
        )

        if round_done:
            if len(active) <= 1:
                # Everyone is all-in (or folded) — no more betting possible
                hand.status = "complete"
                hand.action_on_username = None
                dealer_prompt = "All players are all-in. No more betting. Reveal remaining cards and determine the winner."
            elif hand.current_round == "river":
                hand.status = "complete"
                hand.action_on_username = None
                dealer_prompt = ROUND_PROMPTS["river"]
            else:
                hand.status = "round_complete"
                hand.action_on_username = None
                dealer_prompt = ROUND_PROMPTS[hand.current_round]
        else:
            hand.action_on_username = _find_next_actor(
                hand.hand_players, room.players, req.username, room.dealer_username
            )

        await session.commit()
        await session.refresh(hand)
        return _build_hand_response(hand, dealer_prompt)


@app.post("/rooms/{code}/hands/current/next-round", response_model=HandResponse)
async def advance_round(code: str):
    """Dealer calls this after dealing community cards to open the next betting round."""
    async with SessionLocal() as session:
        room = await _get_room(session, code)

        result = await session.execute(
            select(HandModel).where(
                HandModel.room_id == room.id,
                HandModel.status == "round_complete",
            )
        )
        hand = result.scalar_one_or_none()
        if not hand:
            raise HTTPException(status_code=400, detail="No hand awaiting round advancement")

        current_idx = ROUND_ORDER.index(hand.current_round)
        if current_idx >= len(ROUND_ORDER) - 1:
            hand.status = "complete"
            hand.action_on_username = None
            await session.commit()
            await session.refresh(hand)
            return _build_hand_response(hand, "Hand is complete. Settle the pots.")

        # Advance to next round
        hand.current_round = ROUND_ORDER[current_idx + 1]
        hand.current_bet = 0
        hand.status = "active"

        for hp in hand.hand_players:
            if hp.status == "active":
                hp.bet_this_round = 0
                hp.has_acted = False

        # Check if betting is even possible (need 2+ active players)
        active = [hp for hp in hand.hand_players if hp.status == "active"]
        if len(active) <= 1:
            hand.status = "complete"
            hand.action_on_username = None
            await session.commit()
            await session.refresh(hand)
            return _build_hand_response(
                hand, "All players are all-in. No more betting. Reveal remaining cards and determine the winner."
            )

        first = _find_next_actor(hand.hand_players, room.players, None, room.dealer_username)
        hand.action_on_username = first

        await session.commit()
        await session.refresh(hand)
        return _build_hand_response(hand)


@app.post("/rooms/{code}/hands/{hand_id}/settle", response_model=HandResponse)
async def settle_hand(code: str, hand_id: int, req: SettleRequest):
    """Owner assigns winners for each pot. Pot amounts are added to winners' balances."""
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        if room.owner_username != req.owner_username:
            raise HTTPException(status_code=403, detail="Only the room owner can settle hands")

        result = await session.execute(
            select(HandModel).where(
                HandModel.id == hand_id,
                HandModel.room_id == room.id,
                HandModel.status == "complete",
            )
        )
        hand = result.scalar_one_or_none()
        if not hand:
            raise HTTPException(status_code=404, detail="Hand not found or not ready to settle")

        pots = _compute_pots(hand.hand_players)

        # Validate all pots are accounted for
        assigned_indices = {pw.pot_index for pw in req.pot_winners}
        for i in range(len(pots)):
            if i not in assigned_indices:
                raise HTTPException(status_code=400, detail=f"Missing winner assignment for pot {i}")

        for pw in req.pot_winners:
            if pw.pot_index < 0 or pw.pot_index >= len(pots):
                raise HTTPException(status_code=400, detail=f"Invalid pot index {pw.pot_index}")

            pot = pots[pw.pot_index]
            for winner in pw.winners:
                if winner not in pot.eligible:
                    raise HTTPException(
                        status_code=400,
                        detail=f"'{winner}' is not eligible for pot {pw.pot_index} (eligible: {pot.eligible})",
                    )

            # Split pot among winners, remainder to dealer
            sb = room.small_blind
            num_winners = len(pw.winners)
            base_share = (pot.amount // sb) // num_winners * sb  # largest even split in SB increments
            remainder = pot.amount - base_share * num_winners
            for winner in pw.winners:
                rp = next((p for p in room.players if p.username == winner), None)
                if rp:
                    rp.current_balance += base_share
            if remainder > 0 and room.dealer_username:
                dealer_rp = next((p for p in room.players if p.username == room.dealer_username), None)
                if dealer_rp:
                    dealer_rp.current_balance += remainder

        hand.status = "settled"

        # Auto-rotate dealer to next active player
        active_players = sorted(
            [p for p in room.players if p.is_active],
            key=lambda p: p.position,
        )
        if len(active_players) >= 2 and room.dealer_username:
            usernames = [p.username for p in active_players]
            try:
                current_idx = usernames.index(room.dealer_username)
            except ValueError:
                current_idx = -1
            room.dealer_username = usernames[(current_idx + 1) % len(usernames)]

        await session.commit()
        await session.refresh(hand)
        return _build_hand_response(hand)


# ---------------------------------------------------------------------------
# Transfer endpoints
# ---------------------------------------------------------------------------


@app.post("/rooms/{code}/transfers", response_model=TransferResponse, status_code=201)
async def request_transfer(code: str, req: TransferRequest):
    async with SessionLocal() as session:
        room = await _get_room(session, code)

        if req.from_username == req.to_username:
            raise HTTPException(status_code=400, detail="Cannot transfer to yourself")

        # Verify both players are active in the room
        for uname in (req.from_username, req.to_username):
            result = await session.execute(
                select(PlayerModel).where(
                    PlayerModel.room_id == room.id,
                    PlayerModel.username == uname,
                    PlayerModel.is_active == True,  # noqa: E712
                )
            )
            if not result.scalar_one_or_none():
                raise HTTPException(status_code=400, detail=f"Player '{uname}' is not active in this room")

        transfer = TransferModel(
            room_id=room.id,
            from_username=req.from_username,
            to_username=req.to_username,
            amount=req.amount,
        )
        session.add(transfer)
        await session.commit()
        await session.refresh(transfer)

        return _transfer_response(transfer)


@app.get("/rooms/{code}/transfers", response_model=list[TransferResponse])
async def list_transfers(code: str, status: Optional[str] = Query(None)):
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        stmt = select(TransferModel).where(TransferModel.room_id == room.id)
        if status:
            stmt = stmt.where(TransferModel.status == status)
        stmt = stmt.order_by(TransferModel.created_at.desc())
        result = await session.execute(stmt)
        transfers = result.scalars().all()
        return [_transfer_response(t) for t in transfers]


@app.post("/rooms/{code}/transfers/{transfer_id}/approve", response_model=TransferResponse)
async def approve_transfer(code: str, transfer_id: int, req: OwnerActionRequest):
    return await _resolve_transfer(code, transfer_id, req.owner_username, approved=True)


@app.post("/rooms/{code}/transfers/{transfer_id}/reject", response_model=TransferResponse)
async def reject_transfer(code: str, transfer_id: int, req: OwnerActionRequest):
    return await _resolve_transfer(code, transfer_id, req.owner_username, approved=False)


async def _resolve_transfer(code: str, transfer_id: int, owner_username: str, *, approved: bool) -> TransferResponse:
    async with SessionLocal() as session:
        room = await _get_room(session, code)
        if room.owner_username != owner_username:
            raise HTTPException(status_code=403, detail="Only the room owner can approve/reject transfers")

        result = await session.execute(
            select(TransferModel).where(
                TransferModel.id == transfer_id,
                TransferModel.room_id == room.id,
            )
        )
        transfer = result.scalar_one_or_none()
        if not transfer:
            raise HTTPException(status_code=404, detail="Transfer not found")
        if transfer.status != "pending":
            raise HTTPException(status_code=400, detail=f"Transfer already {transfer.status}")

        if approved:
            # Get sender and receiver
            sender_result = await session.execute(
                select(PlayerModel).where(
                    PlayerModel.room_id == room.id,
                    PlayerModel.username == transfer.from_username,
                )
            )
            sender = sender_result.scalar_one_or_none()

            receiver_result = await session.execute(
                select(PlayerModel).where(
                    PlayerModel.room_id == room.id,
                    PlayerModel.username == transfer.to_username,
                )
            )
            receiver = receiver_result.scalar_one_or_none()

            if not sender or not receiver:
                raise HTTPException(status_code=400, detail="Sender or receiver no longer in room")

            if sender.current_balance < transfer.amount:
                raise HTTPException(status_code=400, detail="Sender has insufficient balance")

            sender.current_balance -= transfer.amount
            sender.initial_balance -= transfer.amount
            receiver.current_balance += transfer.amount
            receiver.initial_balance += transfer.amount
            transfer.status = "approved"
        else:
            transfer.status = "rejected"

        transfer.resolved_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(transfer)

        return _transfer_response(transfer)


def _transfer_response(t: TransferModel) -> TransferResponse:
    return TransferResponse(
        id=t.id,
        from_username=t.from_username,
        to_username=t.to_username,
        amount=t.amount,
        status=t.status,
        created_at=t.created_at,
        resolved_at=t.resolved_at,
    )
