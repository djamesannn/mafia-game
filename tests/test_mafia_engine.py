import asyncio
import unittest

from mafia_engine import (
    EventCodex,
    EventType,
    LLMChatDelta,
    LlamaJSONEvaluator,
    NeurosymbolicRouter,
    load_bot_profiles,
    role_pool_for_size,
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


class FailingEvaluator:
    async def evaluate_chat_message(self, speaker_id, text, listeners_context):
        raise RuntimeError("llm unavailable")

    async def generate_bot_chat(self, bot, public_context):
        return "unreachable"


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



    def test_human_doc_and_cop_intents_override_bot_math(self) -> None:
        state = build_demo_state(seed=28)
        router = NeurosymbolicRouter(state, evaluator=FakeEvaluator())
        router.set_intent_to_heal(3, timestamp=1.0)
        router.set_intent_to_investigate(2, timestamp=1.0)
        self.assertEqual(router._select_doc_target(), 3)
        self.assertEqual(router._select_cop_check_target(), 2)

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




    def test_chat_batch_logs_llm_exception_without_raising(self) -> None:
        async def scenario() -> None:
            state = build_demo_state(seed=25)
            router = NeurosymbolicRouter(state, evaluator=FailingEvaluator())
            await router.enqueue_chat(4, "this will fail", visible_to=[5])
            await router.process_chat_batch()
            self.assertTrue(any("LLM evaluation failed for P4" in event for event in state.event_log))

        asyncio.run(scenario())

    def test_bot_chat_truncates_at_sentence_boundary(self) -> None:
        long_text = "I suspect P2 because the timing is strange. " + "Extra words " * 40
        truncated = LlamaJSONEvaluator._truncate_at_sentence_boundary(long_text, limit=80)
        self.assertEqual(truncated, "I suspect P2 because the timing is strange.")

        no_punctuation = "word " * 40
        truncated_no_punctuation = LlamaJSONEvaluator._truncate_at_sentence_boundary(no_punctuation, limit=40)
        self.assertTrue(truncated_no_punctuation.endswith("..."))

    def test_profiles_json_loads_persistent_identities(self) -> None:
        profiles = load_bot_profiles()
        self.assertGreaterEqual(len(profiles), 10)
        first = next(iter(profiles.values()))
        self.assertIn("name", first)
        self.assertIn("avatar_base", first)
        self.assertIn("psychotype", first)

    def test_mafia_chat_channel_only_updates_mafia_listeners(self) -> None:
        async def scenario() -> None:
            state = build_demo_state(seed=24)
            router = NeurosymbolicRouter(state, evaluator=FakeEvaluator())
            mafia_before = state.bots[2].suspicion_matrix[1]
            town_before = state.bots[4].suspicion_matrix[1]
            await router.enqueue_chat(1, "private plan", channel="mafia")
            await router.process_chat_batch()
            self.assertGreater(state.bots[2].suspicion_matrix[1], mafia_before)
            self.assertEqual(state.bots[4].suspicion_matrix[1], town_before)

        asyncio.run(scenario())


    def test_role_pool_supports_8_and_12_player_rooms(self) -> None:
        self.assertEqual(len(role_pool_for_size(8)), 8)
        self.assertEqual(len(role_pool_for_size(12)), 12)
        self.assertEqual(sum(1 for role in role_pool_for_size(12) if role is Role.MAFIA), 2)
        with self.assertRaises(ValueError):
            role_pool_for_size(10)

    def test_post_mortem_reveal_punishes_defending_mafia(self) -> None:
        state = build_demo_state(seed=26)
        state.bots[2].empathy_matrix[1] = 0.7
        before = state.bots[3].suspicion_matrix[2]
        EventCodex().apply(state, EventType.DAY_EXILE, target_id=1)
        self.assertGreaterEqual(state.bots[3].suspicion_matrix[2], min(1.0, before + 0.15))

    def test_post_mortem_reveal_punishes_pushing_town(self) -> None:
        state = build_demo_state(seed=27)
        state.bots[2].suspicion_matrix[4] = 0.8
        before = state.bots[3].suspicion_matrix[2]
        EventCodex().apply(state, EventType.DAY_EXILE, target_id=4)
        self.assertGreaterEqual(state.bots[3].suspicion_matrix[2], min(1.0, before + 0.12))

    def test_trial_phase_exiles_only_when_guilty_majority(self) -> None:
        state = build_demo_state(seed=21)
        router = NeurosymbolicRouter(state, evaluator=FakeEvaluator())
        target = router.resolve_nomination_phase({1: 2, 3: 2, 4: 2})
        self.assertEqual(target, 2)
        exiled = router.resolve_trial_phase({1: "guilty", 3: "guilty", 4: "innocent", 5: "innocent", 6: "innocent", 7: "guilty", 8: "guilty"})
        self.assertEqual(exiled, 2)
        self.assertFalse(state.bots[2].is_alive)

    def test_trial_vote_events_shift_observer_matrices(self) -> None:
        state = build_demo_state(seed=22)
        observer = state.bots[3]
        observer.empathy_matrix[4] = 0.5
        before_empathy = observer.empathy_matrix[2]
        EventCodex().apply(state, EventType.TRIAL_VOTE_GUILTY, actor_id=2, target_id=4)
        self.assertLess(observer.empathy_matrix[2], before_empathy)

        observer.suspicion_matrix[5] = 0.7
        before_suspicion = observer.suspicion_matrix[6]
        EventCodex().apply(state, EventType.TRIAL_VOTE_INNOCENT, actor_id=6, target_id=5)
        self.assertGreater(observer.suspicion_matrix[6], before_suspicion)

    def test_human_mafia_intent_heavily_weights_consensus_target(self) -> None:
        state = build_demo_state(seed=23)
        state.bots[1].role = Role.MAFIA
        router = NeurosymbolicRouter(state, evaluator=FakeEvaluator())
        router.set_intent_to_kill(8, timestamp=10.0)
        self.assertEqual(router._select_mafia_consensus_target(), 8)

    def test_demo_round_remains_authoritative_fast_path(self) -> None:
        state = build_demo_state(seed=15)
        router = NeurosymbolicRouter(state, evaluator=FakeEvaluator())
        router.resolve_night_phase()
        router._advance_or_finish(Phase.DAY_NOMINATION)
        self.assertIn(state.phase, {Phase.DAY_NOMINATION, Phase.FINISHED})
        if state.winner is not None:
            self.assertIn(state.winner, {Team.TOWN, Team.MAFIA, Team.MANIAC})


if __name__ == "__main__":
    unittest.main()
