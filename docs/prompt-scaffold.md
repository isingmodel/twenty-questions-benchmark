# Prompt Scaffold

## Why This Matters

The benchmark does not ask models to invent their own game strategy from scratch. It gives them a fixed prompting recipe and then compares how different models behave within that recipe.

In the current code, that scaffold is split across four prompt files:

- [`prompts/guesser-initial.txt`](../prompts/guesser-initial.txt)
- [`prompts/guesser-turn.txt`](../prompts/guesser-turn.txt)
- [`prompts/judge-system.txt`](../prompts/judge-system.txt)
- [`prompts/judge-turn-template.txt`](../prompts/judge-turn-template.txt)

The guesser and judge also use different interaction styles. The guesser runs in a persistent multi-turn session. The judge is called statelessly on every turn with the rendered target record and current question.

## Guesser Prompt

The guesser session is initialized with [`prompts/guesser-initial.txt`](../prompts/guesser-initial.txt) as the initial user prompt. The current code does not use a separate guesser system prompt.

That prompt explicitly tells the model to:

- treat the game as a strategic search problem
- ask exactly one question per turn
- start with entity-type questions
- avoid vague or multi-part questions
- switch to a direct identity-check question when sufficiently confident
- treat `Ambiguous` as a cue to repair the wording or entity-type assumption on the next turn

On subsequent turns, [`prompts/guesser-turn.txt`](../prompts/guesser-turn.txt) adds a small fixed wrapper that reminds the model of the previous question and judge reply and then asks for exactly one new question. Because the underlying guesser session is persistent, the model also retains provider-native conversational context from earlier turns.

This is intentional. It reduces prompt drift across runs and makes the benchmark more controlled. It also means the benchmark measures behavior under this fixed scaffold, not raw unprompted search behavior.

## Judge Prompt

The judge uses [`prompts/judge-system.txt`](../prompts/judge-system.txt) as a system prompt and [`prompts/judge-turn-template.txt`](../prompts/judge-turn-template.txt) as the per-turn user prompt.

That prompt explicitly tells the judge to:

- use the target record as the main source of truth
- allow only limited straightforward world knowledge
- prefer `Ambiguous` over a stretched `Yes`
- accept identity checks only for the exact target name or an explicit alias in the record

The turn template renders the full target record as JSON plus the current question and asks the judge to return exactly one compact JSON object with a short reason and one label. The runtime parser normalizes judge outputs back to the three allowed labels: `Yes`, `No`, and `Ambiguous`.

This conservative policy affects turn efficiency and solve rate. Natural-language questions that are slightly underspecified may be penalized more than rigidly phrased questions.

## Reporting Guidance

When writing up results from this repository, describe them as:

- performance under the repository's fixed prompt scaffold and judge policy
- not a prompt-free measure of model capability
- not automatically comparable to benchmarks with different prompting, judge policies, or session semantics
