# Prompt Scaffold

## Why This Matters

The benchmark does not ask models to invent their own game strategy from scratch. It gives them a fixed prompting recipe and then compares how different models behave within that recipe.

## Guesser Prompt

The guesser is initialized with [`prompts/guesser-initial.txt`](../prompts/guesser-initial.txt).

That prompt explicitly tells the model to:

- treat the game as a strategic search problem
- ask exactly one question per turn
- start with entity-type questions
- avoid vague or multi-part questions
- switch to a direct identity-check question when sufficiently confident

This is intentional. It reduces prompt drift across runs and makes the benchmark more controlled. It also means the benchmark measures `model + fixed prompt scaffold`, not raw unprompted search behavior.

## Judge Prompt

The judge uses [`prompts/judge-system.txt`](../prompts/judge-system.txt).

That prompt explicitly tells the judge to:

- use the target record as the main source of truth
- allow only limited straightforward world knowledge
- prefer `Ambiguous` over a stretched `Yes`
- accept identity checks only for the exact target name or an explicit alias

This conservative policy affects turn efficiency and solve rate. Natural-language questions that are slightly underspecified may be penalized more than rigidly phrased questions.

## Reporting Guidance

When writing up results from this repository, describe them as:

- performance under the repository's fixed guesser and judge prompts
- not a prompt-free measure of model capability
- not automatically comparable to benchmarks with different prompting or judge policies
