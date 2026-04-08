from __future__ import annotations

from collections.abc import Callable

from traffic_env.models import IntersectionId, Observation


PolicyFn = Callable[[Observation], str | dict[str, object]]
INTERSECTION_IDS: tuple[IntersectionId, ...] = ("I00", "I01", "I10", "I11")


def always_keep(_: Observation) -> str:
    return "keep"


def alternating_switch(observation: Observation) -> dict[str, object]:
    if any(item.current_phase == "CLEARANCE" for item in observation.intersections.values()):
        return {"action_type": "keep"}
    action = "switch" if observation.step_count % 2 == 0 else "keep"
    return {"action_type": action}


def queue_aware_heuristic(observation: Observation) -> dict[str, object]:
    actions: dict[str, str] = {}
    for iid in INTERSECTION_IDS:
        intersection = observation.intersections[iid]
        if intersection.current_phase == "CLEARANCE":
            actions[iid] = "keep"
            continue
        ns = intersection.queues["N"] + intersection.queues["S"]
        ew = intersection.queues["E"] + intersection.queues["W"]
        current_green_load = ns if intersection.current_phase == "NS_GREEN" else ew
        opposing_load = ew if intersection.current_phase == "NS_GREEN" else ns
        actions[iid] = "switch" if opposing_load > current_green_load + 1 else "keep"
    return {"intersection_actions": actions}


BASELINE_POLICIES: dict[str, PolicyFn] = {
    "always_keep": always_keep,
    "alternating_switch": alternating_switch,
    "queue_aware_heuristic": queue_aware_heuristic,
}
