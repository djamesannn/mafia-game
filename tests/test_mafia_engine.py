import asyncio
import unittest

from mafia_engine import (
    EventCodex,
    EventType,
    LLMChatDelta,
    LlamaJSONEvaluator,
    NeurosymbolicRouter,
    Phase,
    Role,
    Team,
    build_demo_state,
)


class FakeLlama:
    def __init__(self):
        self.last_kwargs = None

    def create_chat_completion(self, **kwargs):
        self.last_kwargs = kwargs
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reasoning_chain":"parsed","d_suspicion":{"P2":0.7},"d_empathy":{"3":-0.7},"stress_impact":0.8}'
                    }
                }
            ]
        }


class FakeEvaluator:
    async def evaluate_chat_message(self, speaker_id, text, listeners_context):
        return LLMChatDelta(
            reasoning_chain="fake evaluator",
            d_suspicion={1: 0.1},
            d_empathy={1: -0.05},
            stress_impact=0.02,
        )

    async def generate_bot_chat(self, bot, public_context):
        return f"P{bot.bot_id} is nervous but watching the room."


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
        router = NeurosymbolicRouter(state, evaluator=FakeEvaluator())
        target_id = router._select_maniac_target()
        self.assertIn(target_id, state.alive_ids(exclude=3))
        self.assertNotEqual(target_id, 3)

    def test_chat_delta_applies_credibility(self) -> None:
        async def scenario() -> None:
            state = build_demo_state(seed=14)
            router = NeurosymbolicRouter(state, evaluator=FakeEvaluator())
            await router.enqueue_chat(4, "P1 is mafia suspicious", visible_to=[5])
            await router.process_chat_batch()
            self.assertGreater(state.bots[5].suspicion_matrix[1], 0.0)

        asyncio.run(scenario())


    def test_llama_evaluator_uses_json_schema_and_clamps_output(self) -> None:
        async def scenario() -> None:
            fake_llm = FakeLlama()
            evaluator = LlamaJSONEvaluator(llm=fake_llm)
            delta = await evaluator.evaluate_chat_message(1, "P2 is dangerous", {"alive_target_ids": [2, 3]})
            self.assertEqual(fake_llm.last_kwargs["response_format"]["schema"], LlamaJSONEvaluator.JSON_SCHEMA)
            self.assertEqual(delta.reasoning_chain, "parsed")
            self.assertEqual(delta.d_suspicion[2], 0.5)
            self.assertEqual(delta.d_empathy[3], -0.5)
            self.assertEqual(delta.stress_impact, 0.5)

        asyncio.run(scenario())

    def test_demo_round_remains_authoritative_fast_path(self) -> None:
        state = build_demo_state(seed=15)
        router = NeurosymbolicRouter(state, evaluator=FakeEvaluator())
        router.resolve_night_phase()
        router._advance_or_finish(Phase.DAY)
        self.assertIn(state.phase, {Phase.DAY, Phase.FINISHED})
        if state.winner is not None:
            self.assertIn(state.winner, {Team.TOWN, Team.MAFIA, Team.MANIAC})


if __name__ == "__main__":
    unittest.main()
