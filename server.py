"""FastAPI/WebSocket real-time Mafia server.

The server owns presentation timing, sockets, human intents, and web actions.
The imported engine remains authoritative for role math, matrix updates, LLM chat
psychology, deaths, role reveals, and win checks. LLM work is scheduled as
background tasks and uses `asyncio.to_thread` inside `LlamaJSONEvaluator`, so the
phase timers continue ticking while inference is running.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from mafia_engine import (
    DAY_NOMINATION_DURATION_SECONDS,
    DAY_TRIAL_DURATION_SECONDS,
    NIGHT_DURATION_SECONDS,
    EventType,
    GameState,
    Gender,
    LlamaJSONEvaluator,
    MafiaBot,
    NeurosymbolicRouter,
    Phase,
    PlayerId,
    Psychotype,
    Role,
    Team,
    first_impression_init,
)

HUMAN_ID: PlayerId = 1
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
PLAYER_NAMES: dict[PlayerId, str] = {
    1: "You",
    2: "Vera",
    3: "Niko",
    4: "Mira",
    5: "Oleg",
    6: "Lina",
    7: "Boris",
    8: "Sasha",
}


class LazyLlamaEvaluator:
    """Defers Qwen GGUF loading until first chat/generation request."""

    def __init__(self) -> None:
        self._delegate: LlamaJSONEvaluator | None = None
        self._lock = asyncio.Lock()

    async def _get_delegate(self) -> LlamaJSONEvaluator:
        if self._delegate is None:
            async with self._lock:
                if self._delegate is None:
                    self._delegate = await asyncio.to_thread(LlamaJSONEvaluator)
        return self._delegate

    async def evaluate_chat_message(self, speaker_id: PlayerId, text: str, listeners_context: dict[str, Any]) -> Any:
        delegate = await self._get_delegate()
        return await delegate.evaluate_chat_message(speaker_id, text, listeners_context)

    async def generate_bot_chat(self, bot: MafiaBot, public_context: dict[str, Any]) -> str:
        delegate = await self._get_delegate()
        return await delegate.generate_bot_chat(bot, public_context)


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for websocket in list(self.active):
            try:
                await websocket.send_json(payload)
            except RuntimeError:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)


class WebMafiaGame:
    """Real-time phase controller for the browser UI."""

    def __init__(self) -> None:
        self.state = self._build_web_state()
        self.router = NeurosymbolicRouter(self.state, evaluator=LazyLlamaEvaluator())
        self.manager = ConnectionManager()
        self.phase_deadline = time.monotonic()
        self.nomination_votes: dict[PlayerId, PlayerId] = {}
        self.trial_votes: dict[PlayerId, str] = {}
        self.chat_tasks: set[asyncio.Task[Any]] = set()
        self.phase_task: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()

    def _build_web_state(self) -> GameState:
        rng = random.Random(88)
        # Player 1 is Mafia so the night "follow the human" intent is visible in the web prototype.
        roles = [Role.MAFIA, Role.BOSS, Role.MANIAC, Role.COP, Role.DOC, Role.WITNESS, Role.CITIZEN, Role.CITIZEN]
        genders: list[Gender] = ["male", "female", "male", "male", "female", "female", "male", "female"]
        bots: dict[PlayerId, MafiaBot] = {}
        for bot_id, (role, gender) in enumerate(zip(roles, genders), start=1):
            bots[bot_id] = MafiaBot(
                bot_id=bot_id,
                gender=gender,
                role=role,
                psychotype=Psychotype(
                    stubbornness=rng.uniform(0.2, 0.8),
                    conformity=rng.uniform(0.2, 0.8),
                    aggression=rng.uniform(0.2, 0.8),
                ),
            )
        state = GameState(bots=bots, rng=rng)
        first_impression_init(state)
        return state

    def start(self) -> None:
        if self.phase_task is None or self.phase_task.done():
            self.phase_task = asyncio.create_task(self.phase_loop())

    async def phase_loop(self) -> None:
        while self.state.phase is not Phase.FINISHED:
            await self.run_night()
            if self._finish_or_set(Phase.DAY_NOMINATION):
                break
            await self.run_nomination()
            if self._finish_or_set(Phase.DAY_TRIAL):
                break
            await self.run_trial()
            if self._finish_or_set(Phase.NIGHT):
                break
        await self.broadcast_state("Game finished")

    async def run_night(self) -> None:
        self.state.phase = Phase.NIGHT
        self.nomination_votes.clear()
        self.trial_votes.clear()
        self.state.trial_target = None
        self.phase_deadline = time.monotonic() + NIGHT_DURATION_SECONDS
        await self.broadcast_state("Night is passing...")

        start = time.monotonic()
        human = self.state.bots[HUMAN_ID]
        human_is_mafia = human.is_alive and human.team is Team.MAFIA
        while True:
            elapsed = time.monotonic() - start
            await self._drain_chat_nonblocking()
            await self.broadcast_state()
            if human_is_mafia:
                has_intent = self.router.intent_to_kill is not None
                if elapsed >= 10.0 and has_intent:
                    break
                if elapsed >= 15.0:
                    break
            elif elapsed >= 3.0:
                break
            await asyncio.sleep(1.0)

        self.router.queue_night_actions()
        report = self.router.apply_morning_report()
        self.router.clear_intent_to_kill()
        parts = ["Morning report ready."]
        if report is not None:
            if report.deaths:
                parts.extend(f"P{dead_id} died and was revealed as {self.state.bots[dead_id].role.value}." for dead_id in report.deaths)
            else:
                parts.append("No one died.")
            if report.witness_observed_killer is not None:
                parts.append(f"Witness revealed P{report.witness_observed_killer}.")
        await self.broadcast_state(" ".join(parts))

    async def run_nomination(self) -> None:
        self.state.phase = Phase.DAY_NOMINATION
        self.phase_deadline = time.monotonic() + DAY_NOMINATION_DURATION_SECONDS
        await self.broadcast_state("Day nomination started. Choose a player to nominate.")
        await asyncio.sleep(1.0)
        await self.cast_bot_nominations()
        while time.monotonic() < self.phase_deadline and self.state.phase is Phase.DAY_NOMINATION:
            await self._drain_chat_nonblocking()
            await self.broadcast_state()
            await asyncio.sleep(1.0)
        self.finalize_nomination()
        await self.broadcast_state(f"P{self.state.trial_target} is on trial." if self.state.trial_target else "No trial target.")

    async def run_trial(self) -> None:
        self.state.phase = Phase.DAY_TRIAL
        self.phase_deadline = time.monotonic() + DAY_TRIAL_DURATION_SECONDS
        await self.broadcast_state("Trial started. Vote Guilty or Acquit.")
        await asyncio.sleep(1.0)
        await self.cast_bot_trial_votes()
        while time.monotonic() < self.phase_deadline and self.state.phase is Phase.DAY_TRIAL:
            await self._drain_chat_nonblocking()
            await self.broadcast_state()
            await asyncio.sleep(1.0)
        exiled = self.finalize_trial()
        if exiled is None:
            await self.broadcast_state("Trial ended with an acquittal.")
        else:
            await self.broadcast_state(f"P{exiled} was exiled and revealed as {self.state.bots[exiled].role.value}.")

    async def cast_bot_nominations(self) -> None:
        async with self._lock:
            for bot in self.state.alive_bots():
                if bot.bot_id == HUMAN_ID or bot.is_frozen or bot.bot_id in self.nomination_votes:
                    continue
                target = self.router._select_day_vote_target(bot)
                if target is not None:
                    self._record_nomination(bot.bot_id, target)

    async def cast_bot_trial_votes(self) -> None:
        async with self._lock:
            target_id = self.state.trial_target
            if target_id is None:
                return
            for bot in self.state.alive_bots():
                if bot.bot_id in {HUMAN_ID, target_id} or bot.bot_id in self.trial_votes:
                    continue
                self._record_trial_vote(bot.bot_id, self.router._select_trial_vote(bot, target_id))

    def finalize_nomination(self) -> None:
        self.router.last_day_votes = self.nomination_votes.copy()
        if not self.nomination_votes:
            self.state.trial_target = None
            self.state.event_log.append("Nomination ended with no valid votes")
            return
        counts = Counter(self.nomination_votes.values())
        max_count = max(counts.values())
        tied = [target_id for target_id, count in counts.items() if count == max_count]
        self.state.trial_target = self.state.rng.choice(tied)
        self.state.event_log.append(f"Player {self.state.trial_target} is on trial")

    def finalize_trial(self) -> PlayerId | None:
        self.router.last_trial_votes = self.trial_votes.copy()
        target_id = self.state.trial_target
        if target_id is None:
            self.state.day_index += 1
            return None
        guilty = sum(1 for vote in self.trial_votes.values() if vote == "guilty")
        innocent = len(self.trial_votes) - guilty
        exiled: PlayerId | None = None
        if guilty > innocent:
            exiled = target_id
            self.router.codex.apply(self.state, EventType.DAY_EXILE, target_id=target_id)
        else:
            self.state.event_log.append(f"Player {target_id} was acquitted")
        self.state.trial_target = None
        self.state.day_index += 1
        return exiled

    async def handle_action(self, player_id: PlayerId, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "chat":
            text = str(message.get("text", "")).strip()
            if text:
                await self.manager.broadcast({"type": "chat", "speaker": player_id, "text": text})
                await self.router.enqueue_chat(player_id, text, visible_to=self.state.alive_ids(exclude=player_id))
                task = asyncio.create_task(self._process_chat_and_broadcast())
                self.chat_tasks.add(task)
                task.add_done_callback(self.chat_tasks.discard)
        elif kind == "nominate":
            await self.human_nominate(player_id, int(message.get("target_id", 0)))
        elif kind == "trial_vote":
            await self.human_trial_vote(player_id, str(message.get("verdict", "innocent")))
        elif kind == "night_action":
            await self.human_night_action(player_id, int(message.get("target_id", 0)))
        await self.broadcast_state()

    async def human_nominate(self, voter_id: PlayerId, target_id: PlayerId) -> None:
        async with self._lock:
            if self.state.phase is not Phase.DAY_NOMINATION:
                return
            if not self._valid_vote_pair(voter_id, target_id):
                return
            self._record_nomination(voter_id, target_id)

    async def human_trial_vote(self, voter_id: PlayerId, verdict: str) -> None:
        async with self._lock:
            if self.state.phase is not Phase.DAY_TRIAL or self.state.trial_target is None:
                return
            if voter_id == self.state.trial_target or voter_id not in self.state.bots or not self.state.bots[voter_id].is_alive:
                return
            vote = "guilty" if verdict == "guilty" else "innocent"
            self._record_trial_vote(voter_id, vote)

    async def human_night_action(self, actor_id: PlayerId, target_id: PlayerId) -> None:
        async with self._lock:
            if self.state.phase is not Phase.NIGHT or not self._valid_vote_pair(actor_id, target_id):
                return
            actor = self.state.bots[actor_id]
            if actor.team is Team.MAFIA:
                self.router.set_intent_to_kill(target_id, time.monotonic())
                self.state.event_log.append(f"P{actor_id} signaled intent to kill P{target_id}")
            elif actor.role is Role.COP:
                self.state.event_log.append(f"P{actor_id} queued an investigation intent on P{target_id}")
            elif actor.role is Role.DOC:
                self.state.event_log.append(f"P{actor_id} queued a heal intent on P{target_id}")

    def _record_nomination(self, voter_id: PlayerId, target_id: PlayerId) -> None:
        self.nomination_votes[voter_id] = target_id
        self.router.codex.apply(self.state, EventType.VOTE_AGAINST, actor_id=voter_id, target_id=target_id)

    def _record_trial_vote(self, voter_id: PlayerId, verdict: str) -> None:
        vote = "guilty" if verdict == "guilty" else "innocent"
        self.trial_votes[voter_id] = vote
        event = EventType.TRIAL_VOTE_GUILTY if vote == "guilty" else EventType.TRIAL_VOTE_INNOCENT
        self.router.codex.apply(self.state, event, actor_id=voter_id, target_id=self.state.trial_target)

    def _valid_vote_pair(self, actor_id: PlayerId, target_id: PlayerId) -> bool:
        return (
            actor_id in self.state.bots
            and target_id in self.state.bots
            and actor_id != target_id
            and self.state.bots[actor_id].is_alive
            and self.state.bots[target_id].is_alive
        )

    async def _process_chat_and_broadcast(self) -> None:
        await self.router.process_chat_batch()
        await self.broadcast_state("Psychological chat deltas applied.")

    async def _drain_chat_nonblocking(self) -> None:
        if self.chat_tasks:
            done = {task for task in self.chat_tasks if task.done()}
            self.chat_tasks -= done

    def _finish_or_set(self, next_phase: Phase) -> bool:
        winner = self.router.check_win_condition()
        if winner is not None:
            self.state.winner = winner
            self.state.phase = Phase.FINISHED
            return True
        self.state.phase = next_phase
        return False

    async def broadcast_state(self, system_message: str | None = None) -> None:
        if system_message:
            await self.manager.broadcast({"type": "system", "text": system_message})
        await self.manager.broadcast({"type": "state", "state": self.snapshot()})

    def snapshot(self) -> dict[str, Any]:
        nomination_counts = Counter(self.nomination_votes.values())
        leading_count = max(nomination_counts.values(), default=0)
        leading_targets = [target_id for target_id, count in nomination_counts.items() if count == leading_count and count > 0]
        trial_counts = Counter(self.trial_votes.values())
        role_counter = Counter(bot.role.value for bot in self.state.alive_bots())
        now = time.monotonic()
        return {
            "phase": self.state.phase.value,
            "timer": max(0, int(self.phase_deadline - now)),
            "day": self.state.day_index,
            "night": self.state.night_index,
            "human_id": HUMAN_ID,
            "human_role": self.state.bots[HUMAN_ID].role.value,
            "winner": self.state.winner.value if self.state.winner else None,
            "trial_target": self.state.trial_target,
            "role_counter": dict(role_counter),
            "nomination_counts": dict(nomination_counts),
            "trial_counts": dict(trial_counts),
            "leading_targets": leading_targets,
            "players": [self._player_snapshot(bot) for bot in self.state.bots.values()],
            "logs": self.state.event_log[-40:],
        }

    def _player_snapshot(self, bot: MafiaBot) -> dict[str, Any]:
        revealed_role = bot.public_role.value if bot.public_role else None
        avatar = f"{bot.avatar_state}.jpg"
        return {
            "id": bot.bot_id,
            "name": PLAYER_NAMES.get(bot.bot_id, f"P{bot.bot_id}"),
            "gender": bot.gender,
            "alive": bot.is_alive,
            "revealed_role": revealed_role,
            "avatar": avatar,
            "vote_count": Counter(self.nomination_votes.values()).get(bot.bot_id, 0),
            "is_leading": bot.bot_id in self.snapshot_leaders(),
            "stress": round(bot.stress_level, 2),
        }

    def snapshot_leaders(self) -> set[PlayerId]:
        counts = Counter(self.nomination_votes.values())
        top = max(counts.values(), default=0)
        return {target_id for target_id, count in counts.items() if count == top and top > 0}


game = WebMafiaGame()
app = FastAPI(title="Neurosymbolic Mafia")


@app.on_event("startup")
async def startup() -> None:
    game.start()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@app.websocket("/ws/{player_id}")
async def websocket_endpoint(websocket: WebSocket, player_id: int) -> None:
    await game.manager.connect(websocket)
    await websocket.send_json({"type": "state", "state": game.snapshot()})
    try:
        while True:
            message = await websocket.receive_json()
            await game.handle_action(player_id, message)
    except WebSocketDisconnect:
        game.manager.disconnect(websocket)
