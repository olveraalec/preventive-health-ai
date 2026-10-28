# Research findings and evidence status

## Research question
Can an agent choose *which information is useful* and *when to trust statistical dependencies*, rather than maximizing memory volume or minimizing covariance risk indiscriminately?

## Findings from controlled synthetic experiments
These are **synthetic, constructed environments**, not clinical validation. Results are copied from the user's terminal output, not independently rerun in this portable package.

| Experiment | Result | Interpretation |
|---|---|---|
| Initial agent benchmark (60 episodes) | static/top-score regret 0.2158; continuous 0.3053; recent-only 0.4242 | Dynamic covariance allocation is not inherently beneficial. Equal weighting and top-score ranking were the same policy here. |
| Aligned covariance (180 episodes) | continuous 0.0640 vs static 0.2586 | Large gain when the simulation deliberately associates lower variance with more useful memories. |
| Uninformative covariance (180 episodes) | continuous = static = 0.2586 | Covariance does not help when it provides no allocation signal. |
| Misleading covariance (180 episodes) | continuous 0.9774 vs static 0.2586 | Strongly adverse consequences of misalignment. |
| Reactive regime gate (540 decisions) | adaptive 0.2358 vs static 0.2586; continuous 0.4333 | Past policy gains can control exposure; delayed reaction at regime transitions. |
| Anticipatory gate (100 synthetic sequences) | anticipatory 0.2604, reactive 0.2658, static 0.2815; mean anticipatory-vs-reactive gain 0.00534; 60% sequence wins; descriptive bootstrap CI [0.00071, 0.01015] | Contextual pre-decision cue can improve performance when the cue was constructed to predict regime. This does not prove discoverability of such cues in real data. |

## Critical validity limitations
- Scenario labels were artificially related to useful memory values and to a synthetic pre-decision cue; outcomes are partly reused across synthetic sequences.
- Reactive simulation updates using *both* policies' counterfactual regret after each decision. In deployed agents, counterfactual results generally are not automatically observable.
- The ensemble's parameter variation is **not** a calibrated predictive uncertainty interval.
- Patient/recording-level generalization and independent shift settings are required.
- Regression metrics (RMSE/MAE/bias/p90 and probabilistic metrics only when genuine sigma exists) apply to supervised forecasting, not directly to synthetic memory utility scores.

## Decision rule to investigate on a new dataset
**Covariance is useful only if adding it improves a downstream held-out objective relative to a covariance-free baseline, conditional on information available before action.** Inspect stability, incremental gain, negative transfer, costs, missingness, transition effects, and false-trust frequency.

## Operational hypothesis
Predict expected incremental gain in advance; permit conservative allocation when expected gain fails to exceed uncertainty and cost thresholds. Keep an explicit static fallback and compare against simpler baselines.
