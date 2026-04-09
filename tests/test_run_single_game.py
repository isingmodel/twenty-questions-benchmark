from __future__ import annotations

import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

from twentyq.episode_runner import (
    ANTHROPIC_PROVIDER,
    JUDGMENT_AMBIGUOUS,
    JUDGMENT_FALSE,
    JUDGMENT_TRUE,
    JudgeDecision,
    OPENAI_PROVIDER,
    PROVIDER,
    FullGameConfig,
    ReasoningConfig,
    QUESTION_TYPE_IDENTITY,
    QUESTION_TYPE_NON_IDENTITY,
    _build_judge_user_prompt,
    _parse_judge_response,
    _reasoning_to_payload,
    _normalize_judge_output,
    _validate_reasoning_config,
    provider_for_model,
    _validate_budget,
    resolve_reasoning_effort,
    run_full_game_episode,
)


class FullGameHelpersTests(unittest.TestCase):
    def test_normalize_judge_output_accepts_common_variants(self) -> None:
        self.assertEqual(_normalize_judge_output(" yes "), JUDGMENT_TRUE)
        self.assertEqual(_normalize_judge_output("YeS!!!"), JUDGMENT_TRUE)
        self.assertEqual(_normalize_judge_output("No."), JUDGMENT_FALSE)
        self.assertEqual(_normalize_judge_output("ambiguous response"), JUDGMENT_AMBIGUOUS)

    def test_normalize_judge_output_rejects_unknown_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unrecognized judge output"):
            _normalize_judge_output("maybe")

    def test_parse_judge_response_accepts_json_with_reason(self) -> None:
        decision = _parse_judge_response(
            '{"label":"Ambiguous","reason":"The target record does not specify EU membership.","question_type":"non_identity","direct_target_guess":false}'
        )
        self.assertEqual(
            decision,
            JudgeDecision(
                label=JUDGMENT_AMBIGUOUS,
                reason="The target record does not specify EU membership.",
                question_type=QUESTION_TYPE_NON_IDENTITY,
                direct_target_guess=False,
            ),
        )

    def test_parse_judge_response_accepts_legacy_label_only_output(self) -> None:
        decision = _parse_judge_response("Yes")
        self.assertEqual(
            decision,
            JudgeDecision(label=JUDGMENT_TRUE, reason=None, question_type=None, direct_target_guess=None),
        )

    def test_parse_judge_response_extracts_json_from_fenced_output(self) -> None:
        decision = _parse_judge_response(
            '```json\n{"label":"No","reason":"The target record says the city is in Europe, not Asia.","question_type":"identity","direct_target_guess":false}\n```'
        )
        self.assertEqual(
            decision,
            JudgeDecision(
                label=JUDGMENT_FALSE,
                reason="The target record says the city is in Europe, not Asia.",
                question_type=QUESTION_TYPE_IDENTITY,
                direct_target_guess=False,
            ),
        )

    def test_parse_judge_response_requires_structured_fields_when_requested(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing question_type"):
            _parse_judge_response('{"label":"Yes","reason":"Correct."}', require_structured=True)

    def test_parse_judge_response_rejects_inconsistent_solved_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "direct target guess as solved without label Yes"):
            _parse_judge_response(
                '{"label":"No","reason":"Wrong candidate.","question_type":"identity","direct_target_guess":true}',
                require_structured=True,
            )

    def test_build_judge_user_prompt_uses_only_current_question(self) -> None:
        prompt = _build_judge_user_prompt(
            judge_turn_template="Hidden target: {{target}}\nCurrent question:\n{{question}}",
            target={"id": "place_paris", "name": "Paris"},
            question="Is it in Europe?",
        )
        self.assertIn('"id": "place_paris"', prompt)
        self.assertIn("Is it in Europe?", prompt)
        self.assertNotIn("Transcript", prompt)
        self.assertNotIn("Judge:", prompt)

    def test_provider_for_model_detects_gemini_openai_and_anthropic(self) -> None:
        self.assertEqual(provider_for_model("gemini-3-flash-preview"), PROVIDER)
        self.assertEqual(provider_for_model("gpt-5"), OPENAI_PROVIDER)
        self.assertEqual(provider_for_model("claude-sonnet-4-20250514"), ANTHROPIC_PROVIDER)

    def test_validate_reasoning_config_rejects_mixed_level_and_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot set both thinking level and thinking budget"):
            _validate_reasoning_config("Guesser", "gemini-3-flash-preview", "low", 128)

    def test_validate_reasoning_config_allows_gemini_25_level(self) -> None:
        reasoning = _validate_reasoning_config("Guesser", "gemini-2.5-flash", "low", None)
        self.assertEqual(reasoning, ReasoningConfig(thinking_level="low"))

    def test_validate_reasoning_config_enforces_gemini_3_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "uses thinking levels"):
            _validate_reasoning_config("Judge", "gemini-3-flash-preview", None, 256)

    def test_validate_reasoning_config_rejects_unsupported_gemini_3_pro_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports thinking levels 'low' and 'high'"):
            _validate_reasoning_config("Guesser", "gemini-3-pro-preview", "medium", None)

    def test_validate_reasoning_config_allows_gemini_31_pro_medium(self) -> None:
        reasoning = _validate_reasoning_config("Guesser", "gemini-3.1-pro-preview", "medium", None)
        self.assertEqual(reasoning, ReasoningConfig(thinking_level="medium"))

    def test_validate_reasoning_config_maps_gpt_thinking_level_to_reasoning_effort(self) -> None:
        reasoning = _validate_reasoning_config("Guesser", "gpt-5", "low", None)
        self.assertEqual(reasoning, ReasoningConfig(reasoning_effort="low"))

    def test_validate_reasoning_config_allows_claude_thinking_budget(self) -> None:
        reasoning = _validate_reasoning_config("Guesser", "claude-sonnet-4-20250514", None, 2048)
        self.assertEqual(reasoning, ReasoningConfig(thinking_budget=2048))

    def test_validate_reasoning_config_allows_claude_opus_46_thinking_budget(self) -> None:
        reasoning = _validate_reasoning_config("Guesser", "claude-opus-4-6", None, 2048)
        self.assertEqual(reasoning, ReasoningConfig(thinking_budget=2048))

    def test_validate_reasoning_config_allows_claude_sonnet_45_thinking_budget(self) -> None:
        reasoning = _validate_reasoning_config("Guesser", "claude-sonnet-4-5", None, 2048)
        self.assertEqual(reasoning, ReasoningConfig(thinking_budget=2048))

    def test_validate_reasoning_config_rejects_claude_35_budget_even_with_new_aliases(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support extended thinking budgets"):
            _validate_reasoning_config("Guesser", "claude-3-5-haiku-20241022", None, 2048)

    def test_validate_reasoning_config_rejects_claude_thinking_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "uses thinking budgets"):
            _validate_reasoning_config("Guesser", "claude-sonnet-4-20250514", "low", None)

    def test_validate_reasoning_config_rejects_claude_budget_on_non_thinking_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support extended thinking budgets"):
            _validate_reasoning_config("Guesser", "claude-3-5-haiku-20241022", None, 2048)

    def test_reasoning_payload_omits_empty_values(self) -> None:
        self.assertEqual(_reasoning_to_payload(ReasoningConfig()), {})
        self.assertEqual(
            _reasoning_to_payload(ReasoningConfig(thinking_level="low")),
            {"thinking_level": "low"},
        )

    def test_resolve_reasoning_effort_maps_gemini_25_flash_to_level(self) -> None:
        reasoning = resolve_reasoning_effort("gemini-2.5-flash", "medium")
        self.assertEqual(reasoning, ReasoningConfig(thinking_level="medium"))

    def test_resolve_reasoning_effort_maps_gemini_25_pro_medium_to_high(self) -> None:
        reasoning = resolve_reasoning_effort("gemini-2.5-pro", "medium")
        self.assertEqual(reasoning, ReasoningConfig(thinking_level="high"))

    def test_resolve_reasoning_effort_maps_gemini_3_flash_to_native_level(self) -> None:
        reasoning = resolve_reasoning_effort("gemini-3-flash-preview", "medium")
        self.assertEqual(reasoning, ReasoningConfig(thinking_level="medium"))

    def test_resolve_reasoning_effort_maps_gemini_3_pro_medium_to_high(self) -> None:
        reasoning = resolve_reasoning_effort("gemini-3-pro-preview", "medium")
        self.assertEqual(reasoning, ReasoningConfig(thinking_level="high"))

    def test_resolve_reasoning_effort_maps_gemini_31_pro_low_to_low(self) -> None:
        reasoning = resolve_reasoning_effort("gemini-3.1-pro-preview", "low")
        self.assertEqual(reasoning, ReasoningConfig(thinking_level="low"))

    def test_resolve_reasoning_effort_maps_gemini_31_flash_lite_low_to_low(self) -> None:
        reasoning = resolve_reasoning_effort("gemini-3.1-flash-lite-preview", "low")
        self.assertEqual(reasoning, ReasoningConfig(thinking_level="low"))

    def test_resolve_reasoning_effort_maps_gemini_31_pro_medium_to_medium(self) -> None:
        reasoning = resolve_reasoning_effort("gemini-3.1-pro-preview", "medium")
        self.assertEqual(reasoning, ReasoningConfig(thinking_level="medium"))

    def test_resolve_reasoning_effort_maps_openai_effort(self) -> None:
        reasoning = resolve_reasoning_effort("gpt-5", "high")
        self.assertEqual(reasoning, ReasoningConfig(reasoning_effort="high"))

    def test_resolve_reasoning_effort_maps_claude_effort_to_budget(self) -> None:
        reasoning = resolve_reasoning_effort("claude-sonnet-4-20250514", "medium")
        self.assertEqual(reasoning, ReasoningConfig(thinking_budget=8192))

    def test_resolve_reasoning_effort_maps_claude_opus_46_effort_to_budget(self) -> None:
        reasoning = resolve_reasoning_effort("claude-opus-4-6", "medium")
        self.assertEqual(reasoning, ReasoningConfig(thinking_budget=8192))

    def test_validate_budget_rejects_non_positive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Budget must be positive"):
            _validate_budget(0)

    def test_run_full_game_does_not_solve_categorical_question_that_mentions_target(self) -> None:
        question = (
            "Is the hidden target one of South Korea's metropolitan cities with a population over 1 million "
            "(such as Busan, Incheon, Daegu, Daejeon, Gwangju, or Ulsan)?"
        )
        judge_output = (
            '{"label":"Yes","reason":"Busan is in that set.","question_type":"non_identity","direct_target_guess":false}'
        )

        class FakeGuesserSession:
            session_mode = "fake-guesser"

        def fake_call_model(_client: object, method_name: str, **kwargs: object) -> tuple[object, int]:
            self.assertEqual(method_name, "generate_turn")
            return ((question, "turn prompt", "req-1", None), 7)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("twentyq.episode_runner._create_client_for_model", return_value=object()),
                patch("twentyq.episode_runner._create_guesser_session", return_value=FakeGuesserSession()),
                patch("twentyq.episode_runner._call_model", side_effect=fake_call_model),
                patch("twentyq.episode_runner._call_stateless_model", return_value=(judge_output, 5)),
            ):
                exit_code, summary = run_full_game_episode(
                    config=FullGameConfig(
                        target_id="place_busan",
                        budget=1,
                        guesser_model="gpt-5.4",
                        judge_model="gpt-5.4-mini",
                        guesser_reasoning=ReasoningConfig(),
                        judge_reasoning=ReasoningConfig(),
                        run_dir=None,
                    ),
                    target={"id": "place_busan", "name": "Busan", "domain": "places", "aliases": []},
                    runs_dir=Path(tmpdir),
                )

        self.assertEqual(exit_code, 0)
        self.assertFalse(summary["solved"])
        self.assertFalse(summary["final_question_correct"])
        self.assertEqual(summary["final_question"], question)

    def test_run_full_game_solves_only_when_judge_marks_direct_target_guess(self) -> None:
        question = "Is the hidden target Busan?"
        judge_output = (
            '{"label":"Yes","reason":"The question directly names the target.","question_type":"identity","direct_target_guess":true}'
        )

        class FakeGuesserSession:
            session_mode = "fake-guesser"

        def fake_call_model(_client: object, method_name: str, **kwargs: object) -> tuple[object, int]:
            self.assertEqual(method_name, "generate_turn")
            return ((question, "turn prompt", "req-1", None), 7)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("twentyq.episode_runner._create_client_for_model", return_value=object()),
                patch("twentyq.episode_runner._create_guesser_session", return_value=FakeGuesserSession()),
                patch("twentyq.episode_runner._call_model", side_effect=fake_call_model),
                patch("twentyq.episode_runner._call_stateless_model", return_value=(judge_output, 5)),
            ):
                exit_code, summary = run_full_game_episode(
                    config=FullGameConfig(
                        target_id="place_busan",
                        budget=1,
                        guesser_model="gpt-5.4",
                        judge_model="gpt-5.4-mini",
                        guesser_reasoning=ReasoningConfig(),
                        judge_reasoning=ReasoningConfig(),
                        run_dir=None,
                    ),
                    target={"id": "place_busan", "name": "Busan", "domain": "places", "aliases": []},
                    runs_dir=Path(tmpdir),
                )

        self.assertEqual(exit_code, 0)
        self.assertTrue(summary["solved"])
        self.assertTrue(summary["final_question_correct"])
        self.assertEqual(summary["final_question"], question)
