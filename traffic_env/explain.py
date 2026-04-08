from __future__ import annotations

from traffic_env.models import Observation


def explain_transition(
    previous: Observation,
    action: str,
    reward: float,
    current: Observation,
) -> dict[str, object]:
    imbalance_before = previous.congestion_imbalance
    imbalance_after = current.congestion_imbalance
    queue_delta = current.total_queue - previous.total_queue
    fairness_delta = current.fairness_index - previous.fairness_index

    if queue_delta < 0:
        action_quality = "Good: total network queue fell after this action."
    elif queue_delta == 0:
        action_quality = "Neutral: total network queue held steady after this action."
    else:
        action_quality = "Costly: total network queue increased after this action."

    imbalance_note = (
        f"Network imbalance changed from {imbalance_before} to {imbalance_after}, "
        f"showing how evenly congestion is spread across the grid."
    )

    reward_note = (
        f"Reward is {reward:.2f} because network congestion, waiting time, fairness, "
        f"and balance were combined with a {action} control decision."
    )
    if current.current_phase == "CLEARANCE":
        reward_note += " The primary intersection is in clearance, so local service paused this step."
    elif fairness_delta > 0.02:
        reward_note += " Fairness improved, which helped the reward."
    elif fairness_delta < -0.02:
        reward_note += " Fairness worsened, which pulled the reward down."

    return {
        "action_quality": action_quality,
        "imbalance_note": imbalance_note,
        "reward_note": reward_note,
        "queue_delta": queue_delta,
        "fairness_delta": round(fairness_delta, 3),
        "regime": current.regime_label,
    }
