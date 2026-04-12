---
title: openenv-traffic-control-benchmark
emoji: "🚦"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Traffic OpenEnv

Traffic OpenEnv is a compact deterministic benchmark for adaptive traffic signal control on a 2x2 network of intersections. It is designed to stay hackathon-safe and easy to run locally while still modeling the coordination problems that appear in real urban traffic systems.

## Why this is useful

Real traffic control is not just a single-junction problem. Neighboring intersections interact through spillback, progression, and uneven demand. A controller that looks good locally can still perform poorly at the network level. This benchmark captures that tradeoff with:

- deterministic 2x2 network dynamics
- per-intersection signal phases and queues
- simplified vehicle propagation between connected intersections
- dense reward and deterministic grading
- lightweight API, frontend, Docker, and inference tooling

## Runtime Requirement

Use Python `3.11` for local execution. The Docker image also uses Python `3.11`.

## Environment Overview

The environment simulates four connected intersections:

- `I00`, `I01`
- `I10`, `I11`

Each intersection has:

- its own `current_phase`
- its own `time_in_phase`
- its own lane queues for `N`, `S`, `E`, `W`
- per-lane wait tracking
- per-intersection switching count
- per-intersection throughput

Supported phases:

- `NS_GREEN`
- `EW_GREEN`
- `CLEARANCE`

Core environment methods:

- `reset(task_name: str | None = None)`
- `step(action)`
- `state()`
- `close()`

## Deterministic Dynamics

At every step the environment:

1. validates the incoming action
2. advances any intersection currently in `CLEARANCE`
3. applies either a broadcast action or per-intersection actions
4. injects deterministic task arrivals
5. serves green approaches at each intersection
6. propagates part of discharged internal flow to neighboring intersections
7. updates waiting-time state
8. computes a dense reward
9. increments `step_count`

There is no randomness.

## Action Space

Two control styles are supported:

Broadcast action for all intersections:

```python
{"action_type": "keep"}
{"action_type": "switch"}
```

Structured multi-action:

```python
{
  "intersection_actions": {
    "I00": "keep",
    "I01": "switch",
    "I10": "keep",
    "I11": "switch"
  }
}
```

For backward compatibility, the existing simple action path still works.

## Observation Space

The observation includes both network-level aggregate fields and detailed per-intersection state.

Top-level fields include:

- `task_name`
- `step_count`
- `max_steps`
- `current_phase`
- `time_in_phase`
- `queues`
- `total_queue`
- `total_wait_time`
- `average_wait_time_per_lane`
- `max_wait_time_per_lane`
- `total_throughput`
- `fairness_index`
- `congestion_imbalance`
- `regime_label`
- `switches_used`
- `intersections`
- `network_dimensions`
- `network_imbalance`
- `last_action_error`
- `done`

`intersections` holds a mapping for `I00`, `I01`, `I10`, and `I11`, each with its own phase, queues, wait metrics, throughput, and switching statistics.

## Reward Design

Reward is dense, deterministic, and network-aware:

```text
congestion_component = 1 / (1 + total_queue / 80)
wait_component = 1 / (1 + average_wait_time / 200)
fairness_component = fairness_index
balance_component = 1 - network_imbalance
base_reward =
    0.40 * congestion_component +
    0.25 * wait_component +
    0.20 * fairness_component +
    0.15 * balance_component
switch_penalty = 0.03 if any intersection switched else 0.0
reward = clamp(base_reward - switch_penalty, 0.0, 1.0)
```

This keeps reward informative even under heavier network congestion while encouraging fairness and balanced operation.

## Metrics

The environment tracks:

- network total queue
- network average wait
- total throughput
- fairness index
- network imbalance
- per-lane average wait
- per-lane max wait
- per-intersection queues, throughput, imbalance, and switches

## Traffic Regimes

The environment classifies each state as:

- `Free Flow`
- `Moderate Congestion`
- `Gridlock`

This is deterministic and based on total network queue.

## Tasks

The benchmark includes deterministic baseline tasks plus adversarial stress tests:

1. `easy_balanced`
   - low balanced arrivals across the network

2. `medium_ew_bias`
   - moderate east-west corridor bias across all intersections

3. `hard_bursty`
   - network-wide shift from north-south demand to east-west burstiness

4. `emergency_priority`
   - strong deterministic pressure on a priority eastbound corridor

5. `arterial_corridor`
   - coordinated east-west progression across connected intersections

6. `downtown_grid_peak`
   - dense downtown peak with high arrivals and local imbalances

7. `incident_blockage`
   - one downstream branch has reduced effective discharge
   - the controller must avoid feeding a bottleneck too aggressively

8. `phase_shift_rush`
   - dominant flow flips sharply multiple times mid-episode
   - the controller must adapt quickly without excessive switching

9. `starvation_trap`
   - heavy corridor demand masks persistent lower-volume side approaches
   - the controller must prevent starvation rather than blindly maximizing throughput

10. `recovery_after_gridlock`
   - the episode begins with seeded queues already present in the network
   - the controller is judged on how well it recovers from a bad initial state

11. `long_horizon_corridor`
   - extended arterial coordination scenario over 48 steps
   - separates stable progression from policies that degrade late in the episode

12. `stability_recovery_cycle`
   - repeated loading, spike, and recovery phases over 52 steps
   - rewards controllers that recover more than once without thrashing

## Stress Tests And Edge Cases

The new adversarial scenarios are designed to make hard tasks genuinely informative:

- `emergency_priority` emphasizes keeping a priority corridor moving
- `incident_blockage` punishes feeding traffic into a constrained branch
- `phase_shift_rush` punishes slow adaptation to regime change
- `starvation_trap` punishes unfair treatment of low-volume approaches
- `recovery_after_gridlock` rewards visible recovery from seeded congestion

These tasks are all deterministic and lightweight, but they expose very different failure modes in network signal control.

The two long-horizon tasks are specifically meant to separate:

- policies that look strong in the first third but collapse later
- policies that recover once but fail on repeated stress
- policies that manage short-term queues by oscillating excessively

## Grading

Each episode is graded deterministically with a normalized scorecard and explicit penalties. Every grade is clamped to `[0.0, 1.0]`.

Core component families:

- congestion
- waiting time
- fairness
- throughput
- stability over time
- sustained fairness
- recovery behavior
- switching efficiency
- priority handling when the task requires it

Normalization strategy:

- congestion combines average queue length and final queue using smooth inverse terms
- waiting time combines average wait and worst observed network wait
- fairness combines the global fairness index with corridor service balance
- throughput is normalized against episode length
- stability blends queue volatility, network imbalance, late-episode queue ratio, and time spent in gridlock
- sustained fairness uses fairness over the full episode rather than only the final step
- recovery behavior uses both recovery quality and recovery time after the worst congestion spike
- switching efficiency rewards productive switching and penalizes phase churn

Representative formulas:

```text
inverse_quality(x, s) = 1 / (1 + x / s)

congestion_quality =
  0.65 * inverse_quality(avg_queue, 45) +
  0.35 * inverse_quality(final_queue, 55)

waiting_quality =
  0.75 * inverse_quality(avg_wait, 220) +
  0.25 * inverse_quality(max_wait, 35)

stability_quality =
  0.40 * inverse_quality(queue_volatility, 9) +
  0.20 * (1 - network_imbalance) +
  0.20 * inverse_quality(max(late_queue_ratio - 1, 0), 0.45) +
  0.20 * (1 - gridlock_fraction)
```

Task-specific weighting:

- `easy_balanced` stays close to the default balanced scorecard
- `medium_ew_bias` slightly increases throughput emphasis
- `hard_bursty` increases stability and switching-efficiency emphasis
- `arterial_corridor` emphasizes throughput and productive switching
- `downtown_grid_peak` emphasizes congestion and waiting-time control
- `emergency_priority` adds a dedicated `priority_handling` component
- `incident_blockage` adds blocked-link management emphasis
- `phase_shift_rush` adds adaptation emphasis
- `starvation_trap` increases fairness emphasis and adds anti-starvation scoring
- `recovery_after_gridlock` increases recovery and congestion emphasis
- `long_horizon_corridor` increases long-horizon stability emphasis
- `stability_recovery_cycle` increases repeated recovery and anti-thrashing emphasis

Protection against degenerate strategies:

- excessive switching triggers a deterministic penalty when switch rate is too high
- starvation triggers a penalty when fairness drops too low
- over-serving one corridor triggers a penalty when service diverges too far from the arrival mix
- late collapse triggers a penalty when the final third of the episode degrades significantly
- oscillation triggers a penalty when switching patterns become unstable

Episode reports:

- every grade returns a `score_breakdown`
- every component includes a short explanation of whether it helped or hurt
- every episode includes an `overall_explanation` describing the main positive driver and the main drag on score
- every episode summary also includes `peak_queue`, `gridlock_steps`, `recovery_time`, `oscillation_count`, and `sustained_fairness`

This makes the benchmark easier to interpret for both judges and model developers: a policy can no longer hide behind a single average queue metric if it is unstable, unfair, or overly switch-heavy.

## Baselines

Three deterministic baseline policies are included:

- `always_keep`
- `alternating_switch`
- `queue_aware_heuristic`

`queue_aware_heuristic` now emits per-intersection actions and is the strongest default benchmark baseline.

Run:

```bash
python3.11 compare_baselines.py
```

## API

The FastAPI app exposes:

- `GET /`
- `GET /demo`
- `GET /tasks`
- `GET /baselines`
- `POST /reset`
- `POST /step`
- `GET /state`

Start locally:

```bash
python3.11 -m uvicorn app:app --reload
```

Example reset:

```bash
curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"arterial_corridor"}'
```

Example broadcast step:

```bash
curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"switch"}'
```

Example structured multi-action step:

```bash
curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"intersection_actions":{"I00":"keep","I01":"switch","I10":"keep","I11":"switch"}}'
```

## Frontend Demo

The dashboard is still served at:

```bash
open http://127.0.0.1:8000/demo
```

It now includes:

- the existing primary intersection control view
- a compact 2x2 network panel
- metrics, explainability, score breakdown, charts, baseline comparison, and long-horizon trend markers

## Inference

`inference.py` remains root-level and runnable:

```bash
python3.11 inference.py
```

It:

- initializes an OpenAI client
- reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, and optional `LOCAL_IMAGE_NAME`
- runs all benchmark tasks
- preserves strict `[START]`, `[STEP]`, `[END]` stdout formatting
- writes separated `[SUMMARY]` metrics to `stderr`, including peak queue, gridlock time, recovery time, and oscillation count

Environment variable pattern:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="deterministic-baseline"
export HF_TOKEN="your_token_if_needed"
export LOCAL_IMAGE_NAME="optional_local_image"
python3.11 inference.py
```

Notes:

- `API_BASE_URL` and `MODEL_NAME` have safe defaults in the script
- `HF_TOKEN` has no hardcoded default
- the current baseline policy is deterministic and does not rely on model output, but the OpenAI client is still initialized in a compliant way for benchmark runners

## Docker

Build:

```bash
docker build -t traffic-openenv .
```

Run:

```bash
docker run --rm -p 8000:8000 traffic-openenv
```

If port `8000` is taken:

```bash
docker run --rm -p 8001:8000 traffic-openenv
```
