from __future__ import annotations

import unittest

from twentyq.episode_runner import (
    JUDGMENT_AMBIGUOUS,
    JUDGMENT_FALSE,
    JUDGMENT_TRUE,
    JudgeDecision,
    OPENAI_PROVIDER,
    PROVIDER,
    ReasoningConfig,
    _build_judge_user_prompt,
    _parse_judge_response,
    _reasoning_to_payload,
    _is_identity_question,
    _normalize_judge_output,
    _validate_reasoning_config,
    provider_for_model,
    _validate_budget,
    resolve_reasoning_effort,
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
        decision = _parse_judge_response('{"label":"Ambiguous","reason":"The target record does not specify EU membership."}')
        self.assertEqual(
            decision,
            JudgeDecision(label=JUDGMENT_AMBIGUOUS, reason="The target record does not specify EU membership."),
        )

    def test_parse_judge_response_accepts_legacy_label_only_output(self) -> None:
        decision = _parse_judge_response("Yes")
        self.assertEqual(decision, JudgeDecision(label=JUDGMENT_TRUE, reason=None))

    def test_parse_judge_response_extracts_json_from_fenced_output(self) -> None:
        decision = _parse_judge_response(
            '```json\n{"label":"No","reason":"The target record says the city is in Europe, not Asia."}\n```'
        )
        self.assertEqual(
            decision,
            JudgeDecision(label=JUDGMENT_FALSE, reason="The target record says the city is in Europe, not Asia."),
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

    def test_identity_question_matches_aliases(self) -> None:
        target = {"name": "Marie Curie", "aliases": ["curie", "marie curie"]}
        self.assertTrue(_is_identity_question("Is it Curie?", target))
        self.assertFalse(_is_identity_question("Is it a scientist?", target))

    def test_identity_question_accepts_minor_normalization_for_aliases(self) -> None:
        target = {"name": "Abraham Lincoln", "aliases": ["lincoln"]}
        self.assertTrue(_is_identity_question("Is it LINCOLN?!", target))

    def test_provider_for_model_detects_gemini_and_openai(self) -> None:
        self.assertEqual(provider_for_model("gemini-3-flash-preview"), PROVIDER)
        self.assertEqual(provider_for_model("gpt-5"), OPENAI_PROVIDER)

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

    def test_validate_budget_rejects_non_positive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Budget must be positive"):
            _validate_budget(0)
