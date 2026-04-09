from __future__ import annotations

from collections import deque

from traffic_env.graders import grade_episode
from traffic_env.models import (
    Action,
    IntersectionObservation,
    Observation,
    Reward,
    StepResult,
    TaskSummary,
)
from traffic_env.tasks import DEFAULT_TASK, INTERSECTION_IDS, TASK_DEFINITIONS, TASK_SCHEDULES


LANES = ("N", "S", "E", "W")
OFFSETS = {"I00": (0, 0), "I01": (0, 1), "I10": (1, 0), "I11": (1, 1)}
NEIGHBORS = {
    "I00": {"E": "I01", "S": "I10"},
    "I01": {"W": "I00", "S": "I11"},
    "I10": {"N": "I00", "E": "I11"},
    "I11": {"N": "I01", "W": "I10"},
}
NETWORK_ENTRY_LANES = {
    "I00": ("N", "W"),
    "I01": ("N", "E"),
    "I10": ("S", "W"),
    "I11": ("S", "E"),
}


class TrafficSignalEnv:
    QUEUE_SCALE = 80.0
    WAIT_SCALE = 200.0
    SWITCH_PENALTY = 0.03
    SERVICE_CAPACITY = 2
    PROPAGATION_FRACTION = 0.5
    EPSILON = 1e-2

    def __init__(self) -> None:
        self._task_name = DEFAULT_TASK
        self._max_steps = TASK_DEFINITIONS[self._task_name].max_steps
        self._schedule = TASK_SCHEDULES[self._task_name]
        self.reset(self._task_name)

    def reset(self, task_name: str | None = None) -> Observation:
        selected_task = task_name or DEFAULT_TASK
        if selected_task not in TASK_DEFINITIONS:
            raise ValueError(f"Unknown task: {selected_task}")
        self._task_name = selected_task
        self._max_steps = TASK_DEFINITIONS[selected_task].max_steps
        self._schedule = TASK_SCHEDULES[selected_task]
        self._step_count = 0
        self._done = False
        self._last_action_error: str | None = None
        self._total_reward = 0.0
        self._queue_sum = 0
        self._phase = {iid: "NS_GREEN" for iid in INTERSECTION_IDS}
        self._pending_phase = {iid: None for iid in INTERSECTION_IDS}
        self._time_in_phase = {iid: 0 for iid in INTERSECTION_IDS}
        self._switches_used = {iid: 0 for iid in INTERSECTION_IDS}
        self._throughput = {iid: 0 for iid in INTERSECTION_IDS}
        self._lane_waits = {
            iid: {lane: deque() for lane in LANES} for iid in INTERSECTION_IDS
        }
        self._arrivals_by_lane = {
            iid: {lane: 0 for lane in LANES} for iid in INTERSECTION_IDS
        }
        self._throughput_by_lane = {
            iid: {lane: 0 for lane in LANES} for iid in INTERSECTION_IDS
        }
        self._max_wait_by_lane = {
            iid: {lane: 0 for lane in LANES} for iid in INTERSECTION_IDS
        }
        self._switch_step_history = {iid: [] for iid in INTERSECTION_IDS}
        self._oscillation_count = 0
        self._total_wait_time = 0
        self._initial_total_queue = 0
        self._seed_initial_queues()
        self._queue_history = [self._total_queue()]
        self._fairness_history = []
        self._regime_history = []
        self._peak_queue = self._total_queue()
        self._peak_queue_step = 0
        return self.state()

    def step(self, action: Action | dict[str, object] | str) -> StepResult:
        if self._done:
            self._last_action_error = "Episode already finished."
            return StepResult(observation=self.state(), reward=Reward(value=0.0))

        self._advance_clearance_phases()
        validated_action = self._validate_action(action)
        action_map = self._action_map(validated_action)
        switched_any = False

        for iid, local_action in action_map.items():
            switched = local_action == "switch" and self._phase[iid] != "CLEARANCE"
            if switched:
                if self._switch_step_history[iid] and (self._step_count - self._switch_step_history[iid][-1]) <= 3:
                    self._oscillation_count += 1
                self._switch_step_history[iid].append(self._step_count)
                self._pending_phase[iid] = "EW_GREEN" if self._phase[iid] == "NS_GREEN" else "NS_GREEN"
                self._phase[iid] = "CLEARANCE"
                self._time_in_phase[iid] = 0
                self._switches_used[iid] += 1
                switched_any = True
            else:
                self._time_in_phase[iid] += 1

        arrivals = self._schedule(self._step_count)
        for iid, lane_arrivals in arrivals.items():
            for lane, cars in lane_arrivals.items():
                self._arrivals_by_lane[iid][lane] += cars
                for _ in range(cars):
                    self._lane_waits[iid][lane].append(0)

        moved_between = {
            iid: {lane: 0 for lane in LANES} for iid in INTERSECTION_IDS
        }
        for iid in INTERSECTION_IDS:
            service_capacity = 0 if self._phase[iid] == "CLEARANCE" else self.SERVICE_CAPACITY
            self._serve_intersection(iid, service_capacity, moved_between)

        for iid, lane_counts in moved_between.items():
            for lane, cars in lane_counts.items():
                for _ in range(cars):
                    self._lane_waits[iid][lane].append(0)
                    self._arrivals_by_lane[iid][lane] += 1

        self._update_waits()

        total_queue = self._total_queue()
        self._queue_history.append(total_queue)
        if total_queue > self._peak_queue:
            self._peak_queue = total_queue
            self._peak_queue_step = self._step_count + 1
        average_wait_time = self._total_wait_time / (self._step_count + 1)
        congestion_component = 1.0 / (1.0 + (total_queue / self.QUEUE_SCALE))
        wait_component = 1.0 / (1.0 + (average_wait_time / self.WAIT_SCALE))
        fairness_component = self._fairness_index()
        balance_component = 1.0 - self._network_imbalance()
        base_reward = (
            (0.4 * congestion_component)
            + (0.25 * wait_component)
            + (0.2 * fairness_component)
            + (0.15 * balance_component)
        )
        reward_value = max(
            0.0,
            min(1.0, base_reward - (self.SWITCH_PENALTY if switched_any else 0.0)),
        )
        reward_value = max(self.EPSILON, min(1.0 - self.EPSILON, reward_value))

        self._step_count += 1
        self._queue_sum += total_queue
        self._total_reward += reward_value
        self._fairness_history.append(fairness_component)
        self._regime_history.append(self._regime_label())
        self._done = self._step_count >= self._max_steps
        return StepResult(observation=self.state(), reward=Reward(value=reward_value))

    def state(self) -> Observation:
        intersections = {
            iid: IntersectionObservation(
                intersection_id=iid,
                current_phase=self._phase[iid],
                time_in_phase=self._time_in_phase[iid],
                queues={lane: len(self._lane_waits[iid][lane]) for lane in LANES},
                total_queue=sum(len(self._lane_waits[iid][lane]) for lane in LANES),
                average_wait_time_per_lane=self._average_wait_time_per_lane(iid),
                max_wait_time_per_lane=dict(self._max_wait_by_lane[iid]),
                switches_used=self._switches_used[iid],
                throughput=self._throughput[iid],
                congestion_imbalance=self._intersection_imbalance(iid),
            )
            for iid in INTERSECTION_IDS
        }
        primary = intersections["I00"]
        return Observation(
            task_name=self._task_name,
            step_count=self._step_count,
            max_steps=self._max_steps,
            current_phase=primary.current_phase,
            time_in_phase=primary.time_in_phase,
            queues=self._boundary_queues(),
            total_queue=self._total_queue(),
            total_wait_time=self._total_wait_time,
            average_wait_time_per_lane=self._network_average_waits(),
            max_wait_time_per_lane=self._network_max_waits(),
            total_throughput=sum(self._throughput.values()),
            fairness_index=self._fairness_index(),
            congestion_imbalance=self._network_total_imbalance(),
            regime_label=self._regime_label(),
            switches_used=sum(self._switches_used.values()),
            intersections=intersections,
            network_dimensions=(2, 2),
            network_imbalance=self._network_imbalance(),
            last_action_error=self._last_action_error,
            done=self._done,
        )

    def episode_summary(self) -> TaskSummary:
        steps_completed = max(self._step_count, 1)
        return TaskSummary(
            task_name=self._task_name,
            steps_completed=self._step_count,
            total_reward=self._total_reward,
            average_queue_length=self._queue_sum / steps_completed,
            average_wait_time=self._total_wait_time / steps_completed,
            fairness_index=self._fairness_index(),
            total_throughput=sum(self._throughput.values()),
            switches_used=sum(self._switches_used.values()),
            final_total_queue=self._total_queue(),
            final_total_wait_time=self._total_wait_time,
            network_imbalance=self._network_imbalance(),
            queue_volatility=self._queue_volatility(),
            peak_queue=self._peak_queue,
            gridlock_steps=self._regime_history.count("Gridlock"),
            moderate_congestion_steps=self._regime_history.count("Moderate Congestion"),
            free_flow_steps=self._regime_history.count("Free Flow"),
            recovery_time=self._recovery_time(),
            oscillation_count=self._oscillation_count,
            sustained_fairness=self._sustained_fairness(),
            task_metrics=self._task_metrics(),
            done=self._done,
        )

    def grade(self):
        return grade_episode(self.episode_summary())

    def close(self) -> None:
        return None

    def _validate_action(self, action: Action | dict[str, object] | str) -> Action:
        self._last_action_error = None
        try:
            if isinstance(action, Action):
                return action
            if isinstance(action, str):
                return Action(action_type=action)
            if isinstance(action, dict):
                return Action(**action)
            raise TypeError("Action must be an Action model, dict, or string.")
        except Exception as exc:
            self._last_action_error = str(exc)
            return Action(action_type="keep")

    def _action_map(self, action: Action) -> dict[str, str]:
        if action.intersection_actions:
            validated = {}
            for iid in INTERSECTION_IDS:
                validated[iid] = action.intersection_actions.get(iid, "keep")
            return validated
        fallback = action.action_type or "keep"
        return {iid: fallback for iid in INTERSECTION_IDS}

    def _advance_clearance_phases(self) -> None:
        for iid in INTERSECTION_IDS:
            if self._phase[iid] == "CLEARANCE" and self._pending_phase[iid] is not None:
                self._phase[iid] = self._pending_phase[iid]
                self._pending_phase[iid] = None
                self._time_in_phase[iid] = 0

    def _serve_intersection(
        self,
        iid: str,
        service_capacity: int,
        moved_between: dict[str, dict[str, int]],
    ) -> None:
        if service_capacity == 0:
            return
        active_lanes = ("N", "S") if self._phase[iid] == "NS_GREEN" else ("E", "W")
        for lane in active_lanes:
            lane_capacity = self._lane_service_capacity(iid, lane, service_capacity)
            served = min(lane_capacity, len(self._lane_waits[iid][lane]))
            for car_index in range(served):
                waited = self._lane_waits[iid][lane].popleft()
                self._max_wait_by_lane[iid][lane] = max(self._max_wait_by_lane[iid][lane], waited)
                self._throughput_by_lane[iid][lane] += 1
                self._throughput[iid] += 1
                if lane in NEIGHBORS[iid] and car_index < int(service_capacity * self.PROPAGATION_FRACTION):
                    next_iid = NEIGHBORS[iid][lane]
                    next_lane = lane
                    moved_between[next_iid][next_lane] += 1

    def _update_waits(self) -> None:
        for iid in INTERSECTION_IDS:
            for lane in LANES:
                updated = deque()
                for wait in self._lane_waits[iid][lane]:
                    new_wait = wait + 1
                    updated.append(new_wait)
                    self._total_wait_time += 1
                    self._max_wait_by_lane[iid][lane] = max(self._max_wait_by_lane[iid][lane], new_wait)
                self._lane_waits[iid][lane] = updated

    def _average_wait_time_per_lane(self, iid: str) -> dict[str, float]:
        return {
            lane: round(sum(self._lane_waits[iid][lane]) / len(self._lane_waits[iid][lane]), 2)
            if self._lane_waits[iid][lane]
            else 0.0
            for lane in LANES
        }

    def _network_average_waits(self) -> dict[str, float]:
        aggregated = {}
        for lane in LANES:
            lane_values = []
            for iid in INTERSECTION_IDS:
                lane_values.extend(self._lane_waits[iid][lane])
            aggregated[lane] = round(sum(lane_values) / len(lane_values), 2) if lane_values else 0.0
        return aggregated

    def _network_max_waits(self) -> dict[str, int]:
        return {
            lane: max(self._max_wait_by_lane[iid][lane] for iid in INTERSECTION_IDS)
            for lane in LANES
        }

    def _fairness_index(self) -> float:
        progress = []
        for iid in INTERSECTION_IDS:
            for lane in LANES:
                progress.append(
                    (self._throughput_by_lane[iid][lane] + 1)
                    / (self._arrivals_by_lane[iid][lane] + 1)
                )
        return max(0.0, min(1.0, min(progress) / max(max(progress), 1e-6)))

    def _intersection_imbalance(self, iid: str) -> int:
        ns = len(self._lane_waits[iid]["N"]) + len(self._lane_waits[iid]["S"])
        ew = len(self._lane_waits[iid]["E"]) + len(self._lane_waits[iid]["W"])
        return abs(ns - ew)

    def _network_total_imbalance(self) -> int:
        return sum(self._intersection_imbalance(iid) for iid in INTERSECTION_IDS)

    def _network_imbalance(self) -> float:
        totals = [
            sum(len(self._lane_waits[iid][lane]) for lane in LANES) for iid in INTERSECTION_IDS
        ]
        if max(totals, default=0) == 0:
            return 0.0
        return min(1.0, (max(totals) - min(totals)) / max(max(totals), 1))

    def _boundary_queues(self) -> dict[str, int]:
        totals = {lane: 0 for lane in LANES}
        for iid, allowed_lanes in NETWORK_ENTRY_LANES.items():
            for lane in allowed_lanes:
                totals[lane] += len(self._lane_waits[iid][lane])
        return totals

    def _total_queue(self) -> int:
        return sum(len(self._lane_waits[iid][lane]) for iid in INTERSECTION_IDS for lane in LANES)

    def _regime_label(self) -> str:
        total_queue = self._total_queue()
        if total_queue < 24:
            return "Free Flow"
        if total_queue < 60:
            return "Moderate Congestion"
        return "Gridlock"

    def _lane_service_capacity(self, iid: str, lane: str, default_capacity: int) -> int:
        if self._task_name == "incident_blockage" and iid in {"I01", "I11"} and lane == "W":
            return 1
        return default_capacity

    def _seed_initial_queues(self) -> None:
        if self._task_name != "recovery_after_gridlock":
            self._initial_total_queue = self._total_queue()
            return
        seeded = {
            "I00": {"N": 4, "S": 3, "E": 5, "W": 4},
            "I01": {"N": 3, "S": 4, "E": 4, "W": 5},
            "I10": {"N": 5, "S": 4, "E": 3, "W": 4},
            "I11": {"N": 4, "S": 5, "E": 4, "W": 3},
        }
        for iid, lane_queues in seeded.items():
            for lane, cars in lane_queues.items():
                self._arrivals_by_lane[iid][lane] += cars
                for _ in range(cars):
                    self._lane_waits[iid][lane].append(0)
        self._initial_total_queue = self._total_queue()

    def _task_metrics(self) -> dict[str, float]:
        ns_arrivals = sum(
            self._arrivals_by_lane[iid]["N"] + self._arrivals_by_lane[iid]["S"]
            for iid in INTERSECTION_IDS
        )
        ew_arrivals = sum(
            self._arrivals_by_lane[iid]["E"] + self._arrivals_by_lane[iid]["W"]
            for iid in INTERSECTION_IDS
        )
        ns_served = sum(
            self._throughput_by_lane[iid]["N"] + self._throughput_by_lane[iid]["S"]
            for iid in INTERSECTION_IDS
        )
        ew_served = sum(
            self._throughput_by_lane[iid]["E"] + self._throughput_by_lane[iid]["W"]
            for iid in INTERSECTION_IDS
        )
        total_arrivals = max(ns_arrivals + ew_arrivals, 1)
        total_served = max(ns_served + ew_served, 1)
        metrics: dict[str, float] = {
            "switch_rate": round(sum(self._switches_used.values()) / max(self._step_count, 1), 3),
            "max_network_wait": float(max(self._network_max_waits().values(), default=0)),
            "corridor_service_mismatch": round(
                abs((ns_served / total_served) - (ns_arrivals / total_arrivals)),
                3,
            ),
            "corridor_balance": round(
                1.0 - min(1.0, abs(ns_served - ew_served) / total_served),
                3,
            ),
            "late_queue_ratio": round(self._late_queue_ratio(), 3),
            "recovery_quality": round(self._recovery_quality(), 3),
            "oscillation_rate": round(self._oscillation_count / max(self._step_count, 1), 3),
            "sustained_fairness": round(self._sustained_fairness(), 3),
            "gridlock_fraction": round(self._regime_history.count("Gridlock") / max(self._step_count, 1), 3),
        }
        if self._task_name == "emergency_priority":
            corridor_waits = []
            for iid in ("I00", "I01"):
                corridor_waits.extend(self._lane_waits[iid]["E"])
            metrics["priority_wait"] = round(sum(corridor_waits) / len(corridor_waits), 3) if corridor_waits else 0.0
        if self._task_name == "incident_blockage":
            blocked_queue = len(self._lane_waits["I01"]["W"]) + len(self._lane_waits["I11"]["W"])
            metrics["blocked_queue"] = float(blocked_queue)
        if self._task_name == "starvation_trap":
            starvation_waits = list(self._lane_waits["I10"]["N"]) + list(self._lane_waits["I11"]["S"])
            metrics["starvation_wait"] = round(sum(starvation_waits) / len(starvation_waits), 3) if starvation_waits else 0.0
        if self._task_name == "recovery_after_gridlock":
            if self._initial_total_queue > 0:
                recovery = (self._initial_total_queue - self._total_queue()) / self._initial_total_queue
            else:
                recovery = 1.0
            metrics["recovery_ratio"] = round(max(0.0, min(1.0, recovery)), 3)
            metrics["initial_queue"] = float(self._initial_total_queue)
        if self._task_name == "phase_shift_rush":
            metrics["adaptation_pressure"] = 1.0 - self._network_imbalance()
        return metrics

    def _queue_volatility(self) -> float:
        if len(self._queue_history) < 2:
            return 0.0
        deltas = [
            abs(current - previous)
            for previous, current in zip(self._queue_history, self._queue_history[1:])
        ]
        return round(sum(deltas) / len(deltas), 3)

    def _late_queue_ratio(self) -> float:
        if len(self._queue_history) < 6:
            return 1.0
        window = max(2, len(self._queue_history) // 3)
        early = self._queue_history[1 : 1 + window]
        late = self._queue_history[-window:]
        early_avg = sum(early) / max(len(early), 1)
        late_avg = sum(late) / max(len(late), 1)
        return late_avg / max(early_avg, 1.0)

    def _recovery_time(self) -> int:
        if self._peak_queue <= 0 or self._peak_queue_step >= len(self._queue_history):
            return 0
        target = max(int(self._peak_queue * 0.7), self._initial_total_queue)
        for index in range(self._peak_queue_step, len(self._queue_history)):
            if self._queue_history[index] <= target:
                return max(index - self._peak_queue_step, 0)
        return max(self._step_count - self._peak_queue_step, 0)

    def _recovery_quality(self) -> float:
        if self._peak_queue <= 0:
            return 1.0
        recovery_time = self._recovery_time()
        baseline_window = max(self._step_count - self._peak_queue_step, 1)
        return max(0.0, min(1.0, 1.0 - (recovery_time / baseline_window)))

    def _sustained_fairness(self) -> float:
        if not self._fairness_history:
            return self._fairness_index()
        return round(sum(self._fairness_history) / len(self._fairness_history), 3)
