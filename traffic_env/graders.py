from __future__ import annotations

from traffic_env.models import GradeResult, TaskSummary


EPS = 1e-6


BASE_WEIGHTS = {
    "congestion": 0.23,
    "waiting_time": 0.15,
    "fairness": 0.14,
    "throughput": 0.13,
    "stability": 0.13,
    "sustained_fairness": 0.10,
    "recovery_behavior": 0.07,
    "switching_efficiency": 0.05,
}

TASK_WEIGHT_OVERRIDES = {
    "easy_balanced": {},
    "medium_ew_bias": {"throughput": 0.18, "fairness": 0.16},
    "hard_bursty": {"congestion": 0.26, "stability": 0.14, "switching_efficiency": 0.12},
    "arterial_corridor": {"throughput": 0.20, "switching_efficiency": 0.12, "fairness": 0.14},
    "downtown_grid_peak": {"congestion": 0.30, "waiting_time": 0.20, "throughput": 0.14},
    "emergency_priority": {
        "congestion": 0.22,
        "waiting_time": 0.14,
        "fairness": 0.14,
        "throughput": 0.12,
        "stability": 0.08,
        "switching_efficiency": 0.08,
        "priority_handling": 0.22,
    },
    "incident_blockage": {
        "congestion": 0.24,
        "waiting_time": 0.18,
        "fairness": 0.18,
        "throughput": 0.12,
        "stability": 0.16,
        "switching_efficiency": 0.12,
    },
    "phase_shift_rush": {
        "congestion": 0.22,
        "waiting_time": 0.14,
        "fairness": 0.14,
        "throughput": 0.18,
        "stability": 0.18,
        "switching_efficiency": 0.14,
    },
    "starvation_trap": {
        "congestion": 0.20,
        "waiting_time": 0.18,
        "fairness": 0.28,
        "throughput": 0.12,
        "stability": 0.10,
        "switching_efficiency": 0.12,
    },
    "recovery_after_gridlock": {
        "congestion": 0.30,
        "waiting_time": 0.16,
        "fairness": 0.14,
        "throughput": 0.16,
        "stability": 0.14,
        "recovery_behavior": 0.10,
        "switching_efficiency": 0.10,
    },
    "long_horizon_corridor": {
        "congestion": 0.20,
        "waiting_time": 0.12,
        "fairness": 0.12,
        "throughput": 0.16,
        "stability": 0.18,
        "sustained_fairness": 0.12,
        "recovery_behavior": 0.06,
        "switching_efficiency": 0.04,
    },
    "stability_recovery_cycle": {
        "congestion": 0.18,
        "waiting_time": 0.14,
        "fairness": 0.12,
        "throughput": 0.12,
        "stability": 0.18,
        "sustained_fairness": 0.10,
        "recovery_behavior": 0.12,
        "switching_efficiency": 0.04,
    },
}


def _bounded_inverse(value: float, scale: float) -> float:
    return 0.99 / (1.0 + (max(value, 0.0) / scale))


def _clamp(value: float) -> float:
    return max(0.01, min(0.99, value))


def _strict_unit_interval(value: float, epsilon: float = EPS) -> float:
    return max(epsilon, min(1.0 - epsilon, value))


def _weights_for_task(task_name: str) -> dict[str, float]:
    weights = dict(BASE_WEIGHTS)
    weights.update(TASK_WEIGHT_OVERRIDES.get(task_name, {}))
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _explain_component(label: str, quality: float, contribution: float) -> str:
    if quality >= 0.8:
        tone = "helped strongly"
    elif quality >= 0.6:
        tone = "supported the score"
    elif quality >= 0.4:
        tone = "was mixed"
    else:
        tone = "dragged the score down"
    return f"{label} {tone}; normalized quality={quality:.2f}, weighted contribution={contribution:.3f}."


def _explain_penalty(label: str, penalty: float, reason: str) -> str:
    if penalty <= 0.0:
        return f"{label} did not materially hurt the score."
    return f"{label} reduced the score by {penalty:.3f} because {reason}."


def grade_episode(summary: TaskSummary) -> GradeResult:
    weights = _weights_for_task(summary.task_name)
    metrics = summary.task_metrics
    steps = max(summary.steps_completed, 1)
    switch_rate = metrics.get("switch_rate", summary.switches_used / steps)
    max_wait = metrics.get("max_network_wait", summary.average_wait_time)
    corridor_balance = metrics.get("corridor_balance", summary.fairness_index)
    service_mismatch = metrics.get("corridor_service_mismatch", 0.0)
    late_queue_ratio = metrics.get("late_queue_ratio", 1.0)
    recovery_quality = metrics.get("recovery_quality", 1.0)
    oscillation_rate = metrics.get("oscillation_rate", 0.0)
    gridlock_fraction = metrics.get("gridlock_fraction", 0.0)

    congestion_quality = (
        0.65 * _bounded_inverse(summary.average_queue_length, 45.0)
        + 0.35 * _bounded_inverse(summary.final_total_queue, 55.0)
    )
    waiting_quality = (
        0.75 * _bounded_inverse(summary.average_wait_time, 220.0)
        + 0.25 * _bounded_inverse(max_wait, 35.0)
    )
    fairness_quality = (0.7 * summary.fairness_index) + (0.3 * corridor_balance)
    throughput_quality = _clamp(summary.total_throughput / max(steps * 10.0, 1.0))
    stability_quality = (
        0.40 * _bounded_inverse(summary.queue_volatility, 9.0)
        + 0.20 * (1.0 - summary.network_imbalance)
        + 0.20 * _bounded_inverse(max(late_queue_ratio - 1.0, 0.0), 0.45)
        + 0.20 * (1.0 - gridlock_fraction)
    )
    sustained_fairness_quality = (
        0.65 * summary.sustained_fairness
        + 0.35 * summary.fairness_index
    )
    recovery_behavior_quality = (
        0.55 * recovery_quality
        + 0.45 * _bounded_inverse(summary.recovery_time, 14.0)
    )
    switches_per_flow = summary.total_throughput / max((summary.switches_used + 1) * 8.0, 1.0)
    switching_control_quality = (
        0.45 * _bounded_inverse(switch_rate, 1.25)
        + 0.25 * _bounded_inverse(summary.oscillation_count, 8.0)
        + 0.30 * _bounded_inverse(oscillation_rate, 0.25)
    )
    switching_efficiency_quality = (
        0.6 * switching_control_quality
        + 0.4 * _clamp(switches_per_flow)
    )

    component_quality = {
        "congestion": congestion_quality,
        "waiting_time": waiting_quality,
        "fairness": fairness_quality,
        "throughput": throughput_quality,
        "stability": stability_quality,
        "sustained_fairness": sustained_fairness_quality,
        "recovery_behavior": recovery_behavior_quality,
        "switching_efficiency": switching_efficiency_quality,
    }

    if "priority_handling" in weights:
        component_quality["priority_handling"] = _bounded_inverse(
            metrics.get("priority_wait", 0.0),
            18.0,
        )

    if summary.task_name == "incident_blockage":
        component_quality["blocked_link_management"] = _bounded_inverse(
            metrics.get("blocked_queue", 0.0),
            20.0,
        )
        weights["blocked_link_management"] = 0.10
    elif summary.task_name == "phase_shift_rush":
        component_quality["adaptation"] = _clamp(metrics.get("adaptation_pressure", 0.0))
        weights["adaptation"] = 0.10
    elif summary.task_name == "starvation_trap":
        component_quality["anti_starvation"] = _bounded_inverse(
            metrics.get("starvation_wait", 0.0),
            22.0,
        )
        weights["anti_starvation"] = 0.10
    elif summary.task_name == "recovery_after_gridlock":
        component_quality["recovery"] = _clamp(metrics.get("recovery_ratio", 0.0))
        weights["recovery"] = 0.12

    positive_weight_total = sum(weights.values())
    normalized_weights = {key: value / positive_weight_total for key, value in weights.items()}

    positive_breakdown: dict[str, float] = {}
    component_explanations: dict[str, str] = {}
    label_map = {
        "congestion": "Congestion control",
        "waiting_time": "Waiting time control",
        "fairness": "Fairness",
        "throughput": "Throughput",
        "stability": "Network stability",
        "sustained_fairness": "Sustained fairness",
        "recovery_behavior": "Recovery behavior",
        "switching_efficiency": "Switching efficiency",
        "priority_handling": "Priority handling",
        "blocked_link_management": "Blocked-link management",
        "adaptation": "Demand-shift adaptation",
        "anti_starvation": "Anti-starvation behavior",
        "recovery": "Recovery quality",
    }
    for key, quality in component_quality.items():
        contribution = normalized_weights[key] * _clamp(quality)
        positive_breakdown[key] = contribution
        component_explanations[key] = _explain_component(
            label_map.get(key, key.replace("_", " ").title()),
            _clamp(quality),
            contribution,
        )

    excessive_switch_penalty = 0.10 * _clamp((switch_rate - 1.05) / 0.9)
    starvation_penalty = 0.12 * _clamp((0.58 - summary.fairness_index) / 0.58)
    corridor_overserve_penalty = 0.10 * _clamp(service_mismatch / 0.35)
    collapse_penalty = 0.14 * _clamp((late_queue_ratio - 1.18) / 0.55)
    oscillation_penalty = 0.10 * _clamp((oscillation_rate - 0.12) / 0.3)

    penalties = {
        "excessive_switching_penalty": excessive_switch_penalty,
        "starvation_penalty": starvation_penalty,
        "corridor_overserve_penalty": corridor_overserve_penalty,
        "late_collapse_penalty": collapse_penalty,
        "oscillation_penalty": oscillation_penalty,
    }
    component_explanations["excessive_switching_penalty"] = _explain_penalty(
        "Excessive switching",
        excessive_switch_penalty,
        "phase changes were too frequent relative to episode length",
    )
    component_explanations["starvation_penalty"] = _explain_penalty(
        "Starvation risk",
        starvation_penalty,
        "fairness fell low enough that some approaches were persistently neglected",
    )
    component_explanations["corridor_overserve_penalty"] = _explain_penalty(
        "Corridor over-serving",
        corridor_overserve_penalty,
        "service allocation diverged too far from the underlying arrival mix",
    )
    component_explanations["late_collapse_penalty"] = _explain_penalty(
        "Late-episode collapse",
        collapse_penalty,
        "queue pressure rose substantially in the final third of the episode",
    )
    component_explanations["oscillation_penalty"] = _explain_penalty(
        "Oscillation risk",
        oscillation_penalty,
        "the controller switched or reversed often enough to suggest unstable control",
    )

    raw_score = sum(positive_breakdown.values()) - sum(penalties.values())
    final_score = _strict_unit_interval(raw_score)

    signed_breakdown = {
        **{key: round(value, 4) for key, value in positive_breakdown.items()},
        **{key: round(-value, 4) for key, value in penalties.items()},
    }
    sorted_positive = sorted(positive_breakdown.items(), key=lambda item: item[1], reverse=True)
    sorted_negative = sorted(penalties.items(), key=lambda item: item[1], reverse=True)
    best_component = sorted_positive[0][0] if sorted_positive else "overall_balance"
    main_penalty = sorted_negative[0][0] if sorted_negative and sorted_negative[0][1] > 0 else None
    overall_explanation = (
        f"Final score {final_score:.3f} came primarily from {label_map.get(best_component, best_component)}. "
        + (
            f"The largest drag was {main_penalty.replace('_', ' ')}."
            if main_penalty
            else "No major degenerate-strategy penalty was triggered."
        )
    )

    return GradeResult(
        task_name=summary.task_name,
        score=final_score,
        average_queue_length=summary.average_queue_length,
        average_wait_time=summary.average_wait_time,
        fairness_index=summary.fairness_index,
        total_throughput=summary.total_throughput,
        switches_used=summary.switches_used,
        final_total_queue=summary.final_total_queue,
        network_imbalance=summary.network_imbalance,
        score_breakdown=signed_breakdown,
        component_explanations=component_explanations,
        overall_explanation=overall_explanation,
        details=(
            "Score is a deterministic weighted combination of normalized congestion, waiting time, "
            "fairness, throughput, stability over time, sustained fairness, recovery behavior, and "
            "switching efficiency, with task-specific weighting and explicit penalties for excessive "
            "switching, starvation, corridor over-serving, late collapse, and oscillation."
        ),
    )
