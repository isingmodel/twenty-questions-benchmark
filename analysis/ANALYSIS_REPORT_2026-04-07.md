# Twenty Questions Benchmark Analysis Report

_Date: 2026-04-07_

This report consolidates the benchmark analysis performed on `results/results.csv` and the deeper log-based investigation using run artifacts under `reports/`.

## Scope

Primary artifacts analyzed:
- `results/results.csv`
- run logs and episodes under `reports/`

Primary columns used from `results.csv`:
- `turns_used`
- `final_question`
- `final_question_correct`
- `error`
- `error_type`
- `guesser_w_effort`
- plus core grouping columns such as `target_id`, `guesser_model`, `judge_model`, and `guesser_reasoning_effort`

## Executive Summary

Core judgment:
- This benchmark is measuring **interactive search policy quality** more than general intelligence.
- The strongest models are not merely knowledgeable; they are better at organizing uncertainty, choosing high-information splits, and moving into identity-checks at the right time.
- The most meaningful target difficulties are not necessarily obscure entities, but targets that induce category drift, ontology failure, or overlapping everyday taxonomies.

Top-line conclusions:
1. The top tier is effectively:
   - `claude-opus-4-6_low`
   - `gpt-5.4_low`
   - `gpt-5.4-mini_high`
2. `claude-opus-4-6_low` is the best-balanced performer across solve rate and efficiency.
3. `gpt-5.4-mini_high` materially outperforms `gpt-5.4-mini_low`; higher reasoning effort improved both solve rate and speed.
4. `gemini-3-flash-preview_low` is an aggressive-search model: lower solve rate, but fast when it succeeds.
5. The hardest targets are not hardest because they are obscure, but because they expose search-policy weaknesses.

## Data Quality and Experimental Conditions

Dataset summary:
- Total runs: **979**
- Guesser variants: **7**
- Targets: **7**
- Judge model: **`gpt-5.4-mini` for all runs**
- Errors: **0**
- Error types: **0**

Observed target set:
- `animal_octopus`
- `character_gandalf`
- `food_croissant`
- `object_toothbrush`
- `object_umbrella`
- `person_marie_curie`
- `place_busan`

Interpretation:
- This is a clean dataset.
- Failures are not due to infrastructure instability.
- Failed runs are strategy failures inside the benchmark protocol, not runtime failures.

## Analysis Plan

The analysis proceeded in four phases:

1. **Data hygiene and structure check**
   - Validate rows, columns, model variants, target coverage, judge configuration, and error cleanliness.

2. **Primary statistical analysis**
   - Compare models on solve rate, average turns on success, and overall turns.
   - Compare targets on solve rate and turn burden.
   - Compute pairwise deltas and bootstrap intervals on selected comparisons.

3. **Hypothesis-driven interpretation**
   - Form explicit hypotheses about model behavior, target difficulty, and reasoning-effort effects.
   - Mark each hypothesis as supported, rejected, or partially supported.

4. **Episode-log deep dive**
   - Inspect concrete success and failure trajectories for the hardest targets:
     - `food_croissant`
     - `object_umbrella`
     - `character_gandalf`
   - Identify failure modes in question sequencing and branching.

## Model-Level Summary

### Aggregate performance

| Model | Runs | Solve Rate | Avg Turns / Success | Avg Turns / All |
|---|---:|---:|---:|---:|
| Claude Opus 4.6 low | 140 | 99.29% | 21.13 | 21.55 |
| GPT-5.4 low | 140 | 98.57% | 23.12 | 23.93 |
| GPT-5.4-mini high | 139 | 98.56% | 23.97 | 24.78 |
| Gemini 3.1 Flash Lite low | 140 | 93.57% | 25.05 | 28.59 |
| GPT-5.4-mini low | 140 | 93.57% | 28.24 | 31.56 |
| Claude Sonnet 4.5 low | 140 | 91.43% | 22.79 | 27.69 |
| Gemini 3 Flash low | 140 | 90.71% | 21.87 | 27.26 |

### Interpretation

- **Claude Opus 4.6 low** is the most balanced model in the benchmark.
  - Highest solve rate
  - Best or near-best efficiency
  - Very low budget-exhaustion frequency

- **GPT-5.4 low** and **GPT-5.4-mini high** are effectively top-tier.
  - Their solve-rate difference versus Opus is small enough that this dataset does not support strong claims of clear superiority.

- **Gemini 3 Flash low** is distinctive.
  - It underperforms on solve rate.
  - But when it does solve, it solves quickly.
  - This is the signature of an aggressive but less stable search policy.

- **GPT-5.4-mini low** is the clearest underperformer in search quality among the GPT variants.
  - Lower solve rate
  - Slower successful solves
  - More budget exhaustion

## Pairwise Checks and Effect Interpretation

### Opus vs GPT-5.4 low

Solve-rate difference:
- `claude-opus-4-6_low - gpt-5.4_low = +0.71%p`
- Bootstrap 95% CI: `(-1.43%p, +3.57%p)`

Turns-per-success difference:
- `-1.99 turns`
- Bootstrap 95% CI: `(-4.48, +0.48)`

Interpretation:
- Opus looks better directionally.
- But this dataset does **not** justify a strong claim that Opus is decisively superior to GPT-5.4 low.

### GPT-5.4-mini high vs GPT-5.4-mini low

Solve-rate difference:
- `+4.99%p`
- Bootstrap 95% CI: `(0.69%p, 9.29%p)`

Turns-per-success difference:
- `-4.27 turns`
- Bootstrap 95% CI: `(-7.59, -0.87)`

Interpretation:
- This is real.
- Higher reasoning effort improved not only solve rate but also efficiency.
- In this benchmark, higher effort did not merely make the model slower-and-safer; it made the model ask better questions sooner.

### Gemini 3 Flash vs Gemini 3.1 Flash Lite

Solve-rate difference:
- `-2.86%p` for Gemini 3 Flash versus 3.1 Flash Lite
- Bootstrap 95% CI includes zero

Turns-per-success difference:
- `-3.19 turns`
- Bootstrap 95% CI approximately `(-6.41, 0.00)`

Interpretation:
- Solve-rate advantage for Flash Lite is not conclusive here.
- Efficiency edge for Gemini 3 Flash is suggestive but borderline.
- The pattern still supports the practical interpretation that Gemini 3 Flash is the more aggressive solver profile.

## Target-Level Summary

### Overall target difficulty

| Target | Solve Rate | Avg Turns |
|---|---:|---:|
| `place_busan` | 100.00% | 14.76 |
| `person_marie_curie` | 99.29% | 15.87 |
| `object_toothbrush` | 98.57% | 22.84 |
| `animal_octopus` | 97.14% | 19.79 |
| `object_umbrella` | 92.14% | 39.98 |
| `character_gandalf` | 89.93% | 32.79 |
| `food_croissant` | 88.57% | 39.38 |

### Model spread by target

Observed solve-rate spread across models:
- `place_busan`: `0.00`
- `person_marie_curie`: `0.05`
- `object_toothbrush`: `0.10`
- `animal_octopus`: `0.15`
- `character_gandalf`: `0.25`
- `object_umbrella`: `0.30`
- `food_croissant`: `0.40`

Interpretation:
- `Busan` and `Marie Curie` are easy and low-separation targets.
- `Croissant`, `Umbrella`, and `Gandalf` are high-separation targets.
- These are the benchmark’s most informative items for distinguishing search policy quality.

## Hypotheses, Validation, and Status

### H1. The top-performing models will dominate in both solve rate and efficiency.

Result:
- **Partially supported**

Evidence:
- The top solve-rate tier is clearly Opus / GPT-5.4 / GPT-5.4-mini high.
- But efficiency is not equally shared within that tier.
- Opus is the cleanest all-around leader.

### H2. Increasing reasoning effort raises solve rate but may cost extra turns.

Result:
- **Rejected**

Evidence:
- For GPT-5.4-mini, higher reasoning effort improved solve rate **and** reduced turns.

Replacement hypothesis:
- Higher reasoning effort can improve the search policy itself, not just confidence calibration.

### H3. Some models will trade solve rate for speed, producing an aggressive-search profile.

Result:
- **Supported**

Evidence:
- Gemini 3 Flash low has lower solve rate but very strong turns-per-success.

### H4. Difficulty is uneven across targets, and model differences widen on specific targets.

Result:
- **Strongly supported**

Evidence:
- Wide target difficulty spread
- Large model spread on Croissant / Umbrella / Gandalf

### H5. Failures are strategy failures, not infrastructure failures.

Result:
- **Supported**

Evidence:
- No error rows
- All failures are effectively budget exhaustion at 80 turns

### H6. Differences among the very top models may not be statistically decisive.

Result:
- **Supported**

Evidence:
- Opus vs GPT-5.4 low differences are directionally suggestive but not statistically decisive in this sample

### H7. The hardest targets are hard because of decomposition difficulty, not pure knowledge scarcity.

Result:
- **Supported**

Evidence:
- `Gandalf`, `Croissant`, and `Umbrella` are not obscure entities.
- Yet they produce large model spread because they induce branching failures and taxonomy drift.

### H8. GPT-5.4-mini high improves because it asks better questions earlier.

Result:
- **Supported**

Evidence:
- Higher solve rate and fewer turns, especially on hard targets

### H9. Final-question surface form is not the main performance driver.

Result:
- **Supported**

Evidence:
- Models differ in identity-check wording style.
- Yet performance does not map cleanly to final-question template choice.
- `solved` and `final_question_correct` match perfectly in all rows.

## Failure Structure

Total unsolved runs: **48**

All unsolved runs reached the budget ceiling:
- `turns_used = 80`

This is important because it means the failure pattern is not “hard crash” or “API instability.”
It is:
- wrong branching
- low-information questioning
- late or missing identity commitment
- inability to recover from an early ontology mistake

Share of runs hitting 80 turns by model:
- Claude Opus 4.6 low: 0.7%
- GPT-5.4 low: 1.4%
- GPT-5.4-mini high: 1.4%
- Gemini 3.1 Flash Lite low: 6.4%
- GPT-5.4-mini low: 6.4%
- Claude Sonnet 4.5 low: 8.6%
- Gemini 3 Flash low: 9.3%

Interpretation:
- Better models do not merely solve more; they avoid entering unrecoverable search loops.

## Deep Dive: Why `Gandalf`, `Croissant`, and `Umbrella` Are Hard

The most important conclusion from the episode logs is this:

> These targets are difficult not because the models lack the facts, but because they expose different failure modes in interactive search.

### 1. `character_gandalf`: Ontology fragility

#### Successful pattern
The successful trajectory is often compact and information-rich:
- person? → Yes
- real or alive? → No
- fictional character? → Yes
- from books? → Yes
- fantasy / Tolkien / wizard? → Yes
- Gandalf? → Yes

This converges quickly when the model correctly enters the **fictional person / literary character** branch.

#### Failure mode
The core failure mode is **entity-typing failure**.

Observed failure patterns include:
- `person?` → No
- `human being, either real or fictional?` → No
- `person?` → Ambiguous

Once that happens, models wander into the wrong ontology:
- object?
- place?
- event?
- concept?
- organism?
- work of art?
- celestial object?
- number?

This is not a lack of knowledge about Gandalf.
It is a failure to stabilize the right category for a fictional being under a conservative judge.

#### Diagnosis
`Gandalf` is hard because it is **ontology-fragile**.
The judge’s interpretation of personhood for fictional beings is sensitive to wording, and a small early miss can derail the whole search tree.

### 2. `food_croissant`: Taxonomy drift

#### Successful pattern
Good runs look like this:
- person? → No
- place? → No
- physical object? → Yes
- food or drink? → Yes
- drink? → No
- prepared dish? → Yes
- baked good / bread / pastry? → Yes
- bread vs pastry? → No
- croissant? → Yes

#### Failure mode
The central failure mode is not ignorance of croissants.
It is **semantic drift into kitchen-adjacent object space**.

Observed failed trajectories include:
- food-related / kitchen-related → Yes
- used for eating? → Yes
- utensil? → No
- plate? → No
- bowl? → No
- final guesses like `kabob pick`, `wafer`, `table runner`

These models end up exploring:
- serving objects
- utensils
- table items
- general kitchen-use ontology
instead of food taxonomy.

#### Log-based lexical signal
Solved Croissant runs mention terms like:
- `food`
- `eat`
- `bread`
- `pastry`
- `baked`

Unsolved runs mention:
- `food`
- `eat`

but under-index on:
- `pastry`
- `baked`
- `bread`

That pattern strongly suggests that the decisive difference is whether the model moves from **food-related** to **food-itself** and then into the right baked taxonomy.

#### Diagnosis
`Croissant` is hard because it is **taxonomy-drift prone**.
It lives too close to kitchen objects, serving items, and ambiguous food categories unless the model deliberately enters pastry taxonomy early.

### 3. `object_umbrella`: Category overlap

#### Successful pattern
Good runs typically converge via:
- object? → Yes
- man-made? → Yes
- portable? → Yes
- worn or carried? → Yes
- worn on body? → No
- umbrella? → Yes

Another successful route is:
- clothing / personal accessory? → Yes
- item of clothing? → No
- umbrella? → Yes

#### Failure mode
The central failure mode is **overlapping category boundaries**.
An umbrella can be framed as:
- portable object
- accessory
- weather gear
- household item
- carried item
- protective item
- tool (depending on wording)

This produces instability in questions like:
- `Is it a tool?`
- `Is it a household item?`
- `Is it used inside a home?`
- `Is it worn or carried?`

Some successful runs receive:
- `tool?` → Ambiguous or No

Some failed runs receive:
- `tool?` → Yes

That is enough to knock the model into the wrong branch.

#### Diagnosis
`Umbrella` is hard because it is **category-overlap heavy**.
The object’s practical function, everyday use, and accessory-like properties sit across multiple weakly overlapping taxonomies, and conservative judging magnifies that instability.

## Log-Level Hypotheses

### L1. `Gandalf` fails when fictional-person typing goes wrong.
- **Supported**

### L2. `Croissant` fails when models stay in kitchen-related object space instead of entering pastry taxonomy.
- **Supported**

### L3. `Umbrella` fails because function-based categories like tool/accessory/household item overlap too much.
- **Supported**

### L4. Successful runs converge by quickly entering a narrow high-information branch, while failed runs spend too long in broad middle categories.
- **Strongly supported**

### L5. Failed runs should have much higher Ambiguous rates.
- **Partially rejected**

Interpretation:
- Ambiguous responses matter, but they are not the main failure driver.
- A larger driver is **wrong-branch persistence** after an early mistake.

## High-Level Interpretation of What the Benchmark Measures

This benchmark behaves more like a test of:
- branch selection
- uncertainty management
- ontology handling
- taxonomy compression
- wrong-branch recovery
- identity-check timing

It behaves less like a simple fact-recall benchmark.

That is why the most informative targets are not the rarest entities.
They are the ones that provoke:
- ontology fragility (`Gandalf`)
- taxonomy drift (`Croissant`)
- category overlap (`Umbrella`)

## Practical Recommendations

### For model comparison
If the goal is stable leaderboard-style evaluation:
- Keep using hard targets with high separation power.
- Do not over-interpret tiny gaps among the top three models.
- Treat `gpt-5.4-mini_high` as evidence that reasoning-effort configuration can materially change benchmark standing.

### For benchmark design
Add more targets that induce distinct reasoning failure modes:
- fictional-person / fictional-being ambiguity
- food-vs-kitchen-object ambiguity
- overlapping everyday object taxonomies

Examples of future useful targets:
- Gandalf-like: fictional beings with human-adjacent roles
- Croissant-like: food items near dish/snack/bread/pastry boundaries
- Umbrella-like: portable objects that overlap tool/accessory/weather gear

### For next-stage metrics
Useful second-order metrics would include:
- early-branch correctness
- time-to-domain-lock
- wrong-branch persistence
- recovery-after-ambiguous
- identity-check timing
- category-jump count

## Final Conclusion

The most important bottom line is:

> The benchmark’s value lies in revealing how models organize the search space under uncertainty.

And the hardest targets reveal different failure pathologies:
- `Gandalf` exposes **ontology fragility**
- `Croissant` exposes **taxonomy drift**
- `Umbrella` exposes **category overlap**

That is why these targets are so useful: they do not just separate models by score; they separate them by *how they fail*.
