import asyncio
import unittest

from mafia_engine import (
    EventCodex,
    EventType,
    NeurosymbolicRouter,
    Phase,
    Role,
    Team,
    build_demo_state,
)


class MafiaEngineTests(unittest.TestCase):
    def test_first_impression_initializes_all_matrix_entries(self) -> None:
        state = build_demo_state(seed=11)
        player_ids = set(state.bots)
        for bot in state.bots.values():
            self.assertEqual(set(bot.suspicion_matrix), player_ids)
            self.assertEqual(set(bot.empathy_matrix), player_ids)
            self.assertEqual(bot.suspicion_matrix[bot.bot_id], 0.0)
            self.assertEqual(bot.empathy_matrix[bot.bot_id], 0.0)

    def test_witness_reveal_forces_citizen_suspicion_and_avatar(self) -> None:
        state = build_demo_state(seed=12)
        EventCodex().apply(state, EventType.WITNESS_REVEAL, target_id=1)
        self.assertEqual(state.bots[1].avatar_state, "boss_male")
        for bot in state.bots.values():
            if bot.role in {Role.CITIZEN, Role.COP, Role.DOC, Role.WITNESS}:
                self.assertEqual(bot.suspicion_matrix[1], 1.0)

    def test_maniac_utility_prefers_balance(self) -> None:
        state = build_demo_state(seed=13)
        router = NeurosymbolicRouter(state)
        target_id = router._select_maniac_target()
        self.assertIn(target_id, state.alive_ids(exclude=3))
        self.assertNotEqual(target_id, 3)

    def test_chat_delta_applies_credibility(self) -> None:
        async def scenario() -> None:
            state = build_demo_state(seed=14)
            router = NeurosymbolicRouter(state)
            await router.enqueue_chat(4, "P1 is mafia suspicious", visible_to=[5])
            await router.process_chat_batch()
            self.assertGreater(state.bots[5].suspicion_matrix[1], 0.0)

        asyncio.run(scenario())

    def test_demo_round_remains_authoritative_fast_path(self) -> None:
        state = build_demo_state(seed=15)
        router = NeurosymbolicRouter(state)
        router.resolve_night_phase()
        router._advance_or_finish(Phase.DAY)
        self.assertIn(state.phase, {Phase.DAY, Phase.FINISHED})
        if state.winner is not None:
            self.assertIn(state.winner, {Team.TOWN, Team.MAFIA, Team.MANIAC})


if __name__ == "__main__":
    unittest.main()
