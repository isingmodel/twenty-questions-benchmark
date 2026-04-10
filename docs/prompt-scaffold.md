# Prompt Scaffold

## Why This Matters

The benchmark does not ask models to invent their own game strategy from scratch. It gives them a fixed prompting recipe and then compares how different models behave within that recipe.

In the current code, that scaffold is split across these prompt files:

- [`prompts/guesser-initial.txt`](../prompts/guesser-initial.txt)
- [`prompts/guesser-turn.txt`](../prompts/guesser-turn.txt)
- [`prompts/guesser-strategic-initial.txt`](../prompts/guesser-strategic-initial.txt)
- [`prompts/judge-system.txt`](../prompts/judge-system.txt)
- [`prompts/judge-turn-template.txt`](../prompts/judge-turn-template.txt)

The guesser and judge also use different interaction styles. The guesser runs in a persistent multi-turn session. The judge is called statelessly on every turn with the rendered target record and current question.

## Guesser Prompt

The guesser session is initialized from a selectable guesser prompt set. The checked-in default uses [`prompts/guesser-initial.txt`](../prompts/guesser-initial.txt) and [`prompts/guesser-turn.txt`](../prompts/guesser-turn.txt). The code also includes a built-in `strategic` preset that swaps only the initial scaffold via [`prompts/guesser-strategic-initial.txt`](../prompts/guesser-strategic-initial.txt) while reusing the same [`prompts/guesser-turn.txt`](../prompts/guesser-turn.txt) turn wrapper. The current code does not use a separate guesser system prompt.

The default prompt explicitly tells the model to:

- treat the game as a strategic search problem
- ask exactly one question per turn
- start with entity-type questions
- avoid vague or multi-part questions
- switch to a direct identity-check question when sufficiently confident
- treat `Ambiguous` as a cue to repair the wording or entity-type assumption on the next turn

On subsequent turns, the chosen guesser turn prompt adds a small fixed wrapper that reminds the model of the previous question and judge reply and then asks for exactly one new question. Because the underlying guesser session is persistent, the model also retains provider-native conversational context from earlier turns.

The `strategic` preset is intentionally more directive. It pushes the model toward:

- maximizing information gain per turn
- using judge-friendly objective wording
- repairing `Ambiguous` answers more directly
- switching to exact identity checks sooner when one candidate dominates

Briefly, [`prompts/guesser-strategic-initial.txt`](../prompts/guesser-strategic-initial.txt) is meant to approximate a strong Twenty Questions policy without becoming overly long. It tells the model to ask constraint-seeking questions that split the remaining candidate set as evenly as possible, prefer stable and directly judgeable properties, and delay exact-name guesses until one candidate is clearly favored. The goal is not richer prose, but a tighter decision rule that should reduce average turns to solve.

References used when writing the `strategic` preset:

- Bellala, Bhavnani, and Scott, "Extensions of Generalized Binary Search to Group Identification and Exponential Costs" (NeurIPS 2010): greedy question selection should reduce expected future queries by splitting hypotheses effectively. https://proceedings.neurips.cc/paper/2010/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html
- Coenen, Nelson, and Gureckis, "Asking the right questions about the psychology of human inquiry: Nine open challenges" (Psychonomic Bulletin & Review, 2019): frames question asking as information search and optimal experiment design. https://pubmed.ncbi.nlm.nih.gov/29869025/
- Ruggeri, Sim, and Xu, "\"Why is Toma late to school again?\" Preschoolers identify the most informative questions" (Developmental Psychology, 2017): contrasts higher-value constraint-seeking questions with weaker hypothesis-scanning strategies. https://pubmed.ncbi.nlm.nih.gov/28661162/
- Denney and Denney, "The Relationship Between Classification and Questioning Strategies Among Adults" (Journal of Gerontology, 1982): older Twenty Questions work linking stronger performance with more constraint-seeking questioning. https://academic.oup.com/geronj/article/37/2/190/520146

For prompt ablations, you can either select a built-in prompt set with `guesser_prompt_set` or provide a custom pair of prompt files through `guesser_initial_prompt_path` and `guesser_turn_prompt_path`.

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
