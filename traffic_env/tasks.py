from __future__ import annotations

from collections.abc import Callable

from traffic_env.models import IntersectionId, TaskDefinition, TaskName


IntersectionArrivals = dict[IntersectionId, dict[str, int]]
ArrivalSchedule = Callable[[int], IntersectionArrivals]
INTERSECTION_IDS: tuple[IntersectionId, ...] = ("I00", "I01", "I10", "I11")


def _uniform_schedule(n: int, s: int, e: int, w: int) -> IntersectionArrivals:
    return {iid: {"N": n, "S": s, "E": e, "W": w} for iid in INTERSECTION_IDS}


def _easy_balanced(_: int) -> IntersectionArrivals:
    return _uniform_schedule(1, 1, 1, 1)


def _medium_ew_bias(_: int) -> IntersectionArrivals:
    return _uniform_schedule(1, 1, 2, 2)


def _hard_bursty(step: int) -> IntersectionArrivals:
    if step <= 9:
        return _uniform_schedule(2, 2, 1, 1)
    if step <= 19:
        return _uniform_schedule(1, 1, 3, 3)
    return _uniform_schedule(2, 2, 2, 2)


def _emergency_priority(step: int) -> IntersectionArrivals:
    if step <= 5:
        return {"I00": {"N": 1, "S": 1, "E": 3, "W": 1}, "I01": {"N": 1, "S": 1, "E": 3, "W": 1}, "I10": {"N": 1, "S": 1, "E": 2, "W": 1}, "I11": {"N": 1, "S": 1, "E": 2, "W": 1}}
    if step <= 11:
        return {"I00": {"N": 1, "S": 1, "E": 4, "W": 2}, "I01": {"N": 1, "S": 1, "E": 4, "W": 2}, "I10": {"N": 1, "S": 1, "E": 3, "W": 2}, "I11": {"N": 1, "S": 1, "E": 3, "W": 2}}
    if step <= 17:
        return {"I00": {"N": 1, "S": 1, "E": 3, "W": 2}, "I01": {"N": 1, "S": 1, "E": 3, "W": 2}, "I10": {"N": 1, "S": 1, "E": 2, "W": 2}, "I11": {"N": 1, "S": 1, "E": 2, "W": 2}}
    return _uniform_schedule(2, 2, 2, 2)


def _incident_blockage(step: int) -> IntersectionArrivals:
    if step <= 9:
        return {
            "I00": {"N": 1, "S": 1, "E": 2, "W": 1},
            "I01": {"N": 1, "S": 1, "E": 1, "W": 3},
            "I10": {"N": 1, "S": 1, "E": 2, "W": 1},
            "I11": {"N": 1, "S": 1, "E": 1, "W": 3},
        }
    if step <= 19:
        return {
            "I00": {"N": 1, "S": 1, "E": 3, "W": 1},
            "I01": {"N": 1, "S": 1, "E": 1, "W": 4},
            "I10": {"N": 1, "S": 1, "E": 3, "W": 1},
            "I11": {"N": 1, "S": 1, "E": 1, "W": 4},
        }
    return _uniform_schedule(2, 2, 2, 2)


def _phase_shift_rush(step: int) -> IntersectionArrivals:
    if step <= 7:
        return _uniform_schedule(2, 2, 1, 1)
    if step <= 15:
        return _uniform_schedule(1, 1, 4, 4)
    if step <= 23:
        return _uniform_schedule(3, 3, 1, 1)
    return _uniform_schedule(2, 2, 2, 2)


def _starvation_trap(step: int) -> IntersectionArrivals:
    if step <= 11:
        return {
            "I00": {"N": 1, "S": 1, "E": 4, "W": 4},
            "I01": {"N": 1, "S": 1, "E": 4, "W": 4},
            "I10": {"N": 3, "S": 1, "E": 4, "W": 4},
            "I11": {"N": 1, "S": 3, "E": 4, "W": 4},
        }
    return {
        "I00": {"N": 1, "S": 1, "E": 3, "W": 3},
        "I01": {"N": 1, "S": 1, "E": 3, "W": 3},
        "I10": {"N": 2, "S": 1, "E": 3, "W": 3},
        "I11": {"N": 1, "S": 2, "E": 3, "W": 3},
    }


def _recovery_after_gridlock(step: int) -> IntersectionArrivals:
    if step <= 7:
        return _uniform_schedule(1, 1, 1, 1)
    if step <= 15:
        return _uniform_schedule(2, 2, 1, 1)
    return _uniform_schedule(1, 1, 2, 2)


def _arterial_corridor(step: int) -> IntersectionArrivals:
    base = {
        "I00": {"N": 1, "S": 1, "E": 3, "W": 1},
        "I01": {"N": 1, "S": 1, "E": 1, "W": 3},
        "I10": {"N": 1, "S": 1, "E": 3, "W": 1},
        "I11": {"N": 1, "S": 1, "E": 1, "W": 3},
    }
    if step >= 12:
        base["I00"]["E"] += 1
        base["I10"]["E"] += 1
    return base


def _downtown_grid_peak(step: int) -> IntersectionArrivals:
    if step <= 9:
        return {
            "I00": {"N": 2, "S": 1, "E": 2, "W": 1},
            "I01": {"N": 1, "S": 2, "E": 1, "W": 2},
            "I10": {"N": 2, "S": 1, "E": 2, "W": 1},
            "I11": {"N": 1, "S": 2, "E": 1, "W": 2},
        }
    if step <= 19:
        return {
            "I00": {"N": 2, "S": 2, "E": 3, "W": 2},
            "I01": {"N": 2, "S": 2, "E": 2, "W": 3},
            "I10": {"N": 3, "S": 2, "E": 2, "W": 2},
            "I11": {"N": 2, "S": 3, "E": 2, "W": 2},
        }
    return {
        "I00": {"N": 2, "S": 2, "E": 2, "W": 2},
        "I01": {"N": 2, "S": 2, "E": 2, "W": 2},
        "I10": {"N": 2, "S": 2, "E": 2, "W": 2},
        "I11": {"N": 2, "S": 2, "E": 2, "W": 2},
    }


def _long_horizon_corridor(step: int) -> IntersectionArrivals:
    if step <= 11:
        return {
            "I00": {"N": 1, "S": 1, "E": 3, "W": 1},
            "I01": {"N": 1, "S": 1, "E": 1, "W": 3},
            "I10": {"N": 1, "S": 1, "E": 3, "W": 1},
            "I11": {"N": 1, "S": 1, "E": 1, "W": 3},
        }
    if step <= 23:
        return {
            "I00": {"N": 2, "S": 1, "E": 4, "W": 1},
            "I01": {"N": 1, "S": 2, "E": 1, "W": 4},
            "I10": {"N": 2, "S": 1, "E": 4, "W": 1},
            "I11": {"N": 1, "S": 2, "E": 1, "W": 4},
        }
    if step <= 35:
        return {
            "I00": {"N": 1, "S": 2, "E": 3, "W": 2},
            "I01": {"N": 2, "S": 1, "E": 2, "W": 3},
            "I10": {"N": 1, "S": 2, "E": 3, "W": 2},
            "I11": {"N": 2, "S": 1, "E": 2, "W": 3},
        }
    return _uniform_schedule(2, 2, 2, 2)


def _stability_recovery_cycle(step: int) -> IntersectionArrivals:
    if step <= 9:
        return _uniform_schedule(2, 2, 1, 1)
    if step <= 19:
        return _uniform_schedule(3, 3, 2, 2)
    if step <= 31:
        return _uniform_schedule(1, 1, 4, 4)
    if step <= 43:
        return _uniform_schedule(1, 1, 1, 1)
    return _uniform_schedule(2, 2, 2, 2)


TASK_DEFINITIONS: dict[TaskName, TaskDefinition] = {
    "easy_balanced": TaskDefinition(
        name="easy_balanced",
        description="Balanced low-demand 2x2 grid with steady arrivals at every junction.",
        motivation="Tests whether a network controller can remain stable when demand is light and symmetric.",
        objective="Maintain low queue and low switching under symmetric light demand.",
        stress_note="This is the calibration task and should be the easiest scenario.",
        max_steps=20,
    ),
    "medium_ew_bias": TaskDefinition(
        name="medium_ew_bias",
        description="Grid-wide east-west commuter bias with lighter north-south demand.",
        motivation="Rewards controllers that identify corridor bias while still preserving side-street fairness.",
        objective="Prioritize the dominant east-west corridor without over-serving empty directions.",
        stress_note="Controllers that fail to recognize corridor bias will accumulate avoidable queue.",
        max_steps=24,
    ),
    "hard_bursty": TaskDefinition(
        name="hard_bursty",
        description="Grid-wide demand shift from north-south bias to stronger east-west burstiness and then heavy balance.",
        motivation="Tests adaptation under changing network conditions and inter-junction spillover pressure.",
        objective="Adapt quickly to shifting directional demand while avoiding oscillatory switching.",
        stress_note="A static policy will be punished when the dominant flow flips mid-episode.",
        max_steps=30,
    ),
    "emergency_priority": TaskDefinition(
        name="emergency_priority",
        description="Priority eastbound corridor through the grid with sustained emergency-style pressure.",
        motivation="Represents a real operational need to protect one corridor without starving the rest of the network.",
        objective="Keep eastbound corridor delay low while preserving minimum fairness elsewhere.",
        stress_note="The grader weighs corridor delay more heavily than in normal tasks.",
        max_steps=24,
    ),
    "incident_blockage": TaskDefinition(
        name="incident_blockage",
        description="One eastern downstream link is partially blocked, reducing effective discharge on that branch of the network.",
        motivation="Models a lane-blocking incident where naive signal timing will feed traffic directly into a bottleneck.",
        objective="Prevent blocked approaches from overflowing while rerouting pressure through the rest of the grid.",
        stress_note="Controllers are punished if they keep feeding the blocked movement too aggressively.",
        max_steps=28,
    ),
    "phase_shift_rush": TaskDefinition(
        name="phase_shift_rush",
        description="The dominant direction flips sharply twice during the episode, simulating school release and commuter surges.",
        motivation="Measures how quickly a policy can pivot after a major regime change.",
        objective="Respond to abrupt demand reversals without wasting too many steps in clearance.",
        stress_note="Policies that adapt too slowly will look good early and fail badly later.",
        max_steps=28,
    ),
    "starvation_trap": TaskDefinition(
        name="starvation_trap",
        description="Heavy corridor demand masks a smaller but persistent side-street stream that becomes unfairly delayed under naive control.",
        motivation="Highlights the need for fairness-aware traffic control rather than queue-only prioritization.",
        objective="Prevent starvation on low-volume approaches while still controlling the dominant corridor.",
        stress_note="A naive throughput-seeking controller will often ignore one approach until max waits explode.",
        max_steps=26,
    ),
    "recovery_after_gridlock": TaskDefinition(
        name="recovery_after_gridlock",
        description="The episode starts with preloaded queues and the controller must recover to a stable operating regime.",
        motivation="Tests whether a controller can unwind an already congested network instead of only reacting to fresh arrivals.",
        objective="Reduce the initial backlog quickly and finish with a visibly healthier network state.",
        stress_note="This task rewards improvement from a bad starting point, not just low steady-state delay.",
        max_steps=26,
    ),
    "arterial_corridor": TaskDefinition(
        name="arterial_corridor",
        description="A directional arterial running across the 2x2 network that benefits from coordinated east-west progression.",
        motivation="Measures whether the policy can support coordinated movement across connected intersections.",
        objective="Maintain progression on the arterial while limiting spillback into side streets.",
        stress_note="Coordination matters more here than isolated local optimality.",
        max_steps=28,
    ),
    "downtown_grid_peak": TaskDefinition(
        name="downtown_grid_peak",
        description="Dense downtown peak with high arrivals at every junction and shifting local imbalances.",
        motivation="Stresses fairness, throughput, and network imbalance under saturated urban conditions.",
        objective="Maximize throughput while keeping network imbalance under control in saturation.",
        stress_note="This is the highest-load stress test and should feel near-gridlock throughout.",
        max_steps=32,
    ),
    "long_horizon_corridor": TaskDefinition(
        name="long_horizon_corridor",
        description="A longer coordinated arterial scenario that rewards sustained progression instead of short-term queue chasing.",
        motivation="Separates policies that can hold a corridor together for many steps from those that look good early and unravel later.",
        objective="Maintain corridor flow and fairness over an extended horizon without collapsing late in the episode.",
        stress_note="Long-horizon instability and late-episode collapse are graded explicitly here.",
        max_steps=48,
    ),
    "stability_recovery_cycle": TaskDefinition(
        name="stability_recovery_cycle",
        description="A long episode with repeated loading, congestion spike, and recovery phases across the grid.",
        motivation="Measures whether a controller can recover from stress repeatedly rather than only surviving one transient surge.",
        objective="Absorb congestion spikes, recover quickly, and keep the network stable through the full cycle.",
        stress_note="Policies that oscillate or recover too slowly will score poorly even if the early episode looks strong.",
        max_steps=52,
    ),
}

TASK_SCHEDULES: dict[TaskName, ArrivalSchedule] = {
    "easy_balanced": _easy_balanced,
    "medium_ew_bias": _medium_ew_bias,
    "hard_bursty": _hard_bursty,
    "emergency_priority": _emergency_priority,
    "incident_blockage": _incident_blockage,
    "phase_shift_rush": _phase_shift_rush,
    "starvation_trap": _starvation_trap,
    "recovery_after_gridlock": _recovery_after_gridlock,
    "arterial_corridor": _arterial_corridor,
    "downtown_grid_peak": _downtown_grid_peak,
    "long_horizon_corridor": _long_horizon_corridor,
    "stability_recovery_cycle": _stability_recovery_cycle,
}

DEFAULT_TASK: TaskName = "easy_balanced"


def get_task_names() -> list[TaskName]:
    return list(TASK_DEFINITIONS.keys())
