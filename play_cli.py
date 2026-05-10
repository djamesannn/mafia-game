"""Playable terminal loop for the neurosymbolic Mafia backend.

Run with:
    python play_cli.py

The script expects `qwen2.5-3b-instruct-q4_k_m.gguf` in the repository root and
`llama-cpp-python` installed with CUDA support. The local model is used only for
slow-path chat evaluation and lightweight bot utterances; `mafia_engine.py`
continues to own all authoritative votes, deaths, role actions, and win checks.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter

from mafia_engine import (
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
BOT_NAMES: dict[PlayerId, str] = {
    1: "You",
    2: "Vera",
    3: "Niko",
    4: "Mira",
    5: "Oleg",
    6: "Lina",
    7: "Boris",
    8: "Sasha",
}


def build_cli_state(seed: int = 41) -> GameState:
    """Create the 8-player CLI setup with player 1 reserved for the human."""

    rng = random.Random(seed)
    roles = [Role.CITIZEN, Role.BOSS, Role.MAFIA, Role.MANIAC, Role.COP, Role.DOC, Role.WITNESS, Role.CITIZEN]
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


def name_of(player_id: PlayerId) -> str:
    return f"{BOT_NAMES.get(player_id, f'P{player_id}')} (P{player_id})"


def print_roster(state: GameState) -> None:
    print("\n=== Roster ===")
    for bot in state.bots.values():
        status = "alive" if bot.is_alive else f"dead/revealed={bot.public_role.value if bot.public_role else '?'}"
        marker = " <- human" if bot.bot_id == HUMAN_ID else ""
        print(f"P{bot.bot_id}: {BOT_NAMES.get(bot.bot_id, 'Bot')} [{status}]{marker}")
    print(f"Your hidden role: {state.bots[HUMAN_ID].role.value}\n")


def print_morning_report(state: GameState, router: NeurosymbolicRouter) -> None:
    report = router.apply_morning_report()
    print("\n=== Morning report ===")
    if report is None:
        print("No night report is available.")
        return
    if not report.deaths:
        print("No one died during the night.")
    for dead_id in report.deaths:
        bot = state.bots[dead_id]
        print(f"{name_of(dead_id)} died. Revealed role: {bot.role.value}; avatar={bot.avatar_state}")
    if report.witness_observed_killer is not None:
        killer = state.bots[report.witness_observed_killer]
        print(f"Witness reveal: {name_of(killer.bot_id)} is publicly exposed as {killer.role.value}.")
    if report.cop_target is not None:
        print(f"A cop investigation occurred: target was P{report.cop_target}. Result remains private to role logic.")


def fallback_bot_sentence(bot: MafiaBot, state: GameState) -> str:
    alive_targets = state.alive_ids(exclude=bot.bot_id)
    if not alive_targets:
        return "I have nothing useful to add."
    target = max(alive_targets, key=lambda player_id: bot.suspicion_matrix[player_id])
    if bot.stress_level > 0.55:
        return f"I am under pressure, but P{target} looks more suspicious to me."
    return f"I am watching P{target}; the pattern feels off."


async def bot_day_chat(state: GameState, router: NeurosymbolicRouter) -> None:
    """Print and enqueue one public message from each living FSM bot."""

    context = {
        "day": state.day_index,
        "alive_ids": state.alive_ids(),
        "recent_events": state.event_log[-6:],
    }
    for bot in state.alive_bots():
        if bot.bot_id == HUMAN_ID:
            continue
        try:
            if bot.stress_level > 0.35:
                text = await router.evaluator.generate_bot_chat(bot, context)
            else:
                text = fallback_bot_sentence(bot, state)
        except Exception as exc:  # CLI should remain playable if generation hiccups.
            text = f"{fallback_bot_sentence(bot, state)} [local generation unavailable: {exc}]"
        print(f"{name_of(bot.bot_id)}: {text}")
        await router.enqueue_chat(bot.bot_id, text, visible_to=state.alive_ids(exclude=bot.bot_id))


def print_vote_results(state: GameState, router: NeurosymbolicRouter, exiled: PlayerId | None) -> None:
    print("\n=== Vote results ===")
    if not router.last_day_votes:
        print("No valid votes were cast.")
        return
    for voter_id, target_id in sorted(router.last_day_votes.items()):
        print(f"{name_of(voter_id)} voted against {name_of(target_id)}")
    counts = Counter(router.last_day_votes.values())
    print("Tally:", ", ".join(f"P{target}: {count}" for target, count in sorted(counts.items())))
    if exiled is None:
        print("No one was exiled.")
    else:
        bot = state.bots[exiled]
        print(f"Exiled: {name_of(exiled)}. Revealed role: {bot.role.value}; avatar={bot.avatar_state}")


async def run_cli_game() -> None:
    print("Loading Qwen-2.5-3B-Instruct via llama-cpp-python/CUDA...")
    evaluator = LlamaJSONEvaluator()
    state = build_cli_state()
    router = NeurosymbolicRouter(state, evaluator=evaluator)
    print_roster(state)

    while state.phase is not Phase.FINISHED:
        if state.phase is Phase.NIGHT:
            print(f"\n=== Night {state.night_index} ===")
            print("Night is passing...")
            router.queue_night_actions()
            router._advance_or_finish(Phase.DAY)
            continue

        print_morning_report(state, router)
        if state.phase is Phase.FINISHED:
            break

        print(f"\n=== Day {state.day_index} ===")
        await bot_day_chat(state, router)

        if state.bots[HUMAN_ID].is_alive:
            message = await asyncio.to_thread(input, "Your message: ")
            if message.strip():
                await router.enqueue_chat(HUMAN_ID, message.strip(), visible_to=state.alive_ids(exclude=HUMAN_ID))
        else:
            print("You are dead. Press Enter to continue observing.")
            await asyncio.to_thread(input, "")

        await router.process_chat_batch()
        exiled = router.resolve_day_phase()
        print_vote_results(state, router, exiled)
        router._advance_or_finish(Phase.NIGHT)

    print("\n=== Game over ===")
    winner = state.winner or router.check_win_condition()
    if winner is Team.TOWN:
        print("Town wins!")
    elif winner is Team.MAFIA:
        print("Mafia wins!")
    elif winner is Team.MANIAC:
        print("Maniac wins!")
    else:
        print("No winner was determined.")
    print_roster(state)


if __name__ == "__main__":
    asyncio.run(run_cli_game())
