from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


Phase = Literal["NS_GREEN", "EW_GREEN", "CLEARANCE"]
ActionType = Literal["keep", "switch"]
TaskName = Literal[
    "easy_balanced",
    "medium_ew_bias",
    "hard_bursty",
    "emergency_priority",
    "incident_blockage",
    "phase_shift_rush",
    "starvation_trap",
    "recovery_after_gridlock",
    "arterial_corridor",
    "downtown_grid_peak",
    "long_horizon_corridor",
    "stability_recovery_cycle",
]
TrafficRegime = Literal["Free Flow", "Moderate Congestion", "Gridlock"]
IntersectionId = Literal["I00", "I01", "I10", "I11"]


class IntersectionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intersection_id: IntersectionId
    current_phase: Phase
    time_in_phase: int = Field(ge=0)
    queues: dict[str, int]
    total_queue: int = Field(ge=0)
    average_wait_time_per_lane: dict[str, float]
    max_wait_time_per_lane: dict[str, int]
    switches_used: int = Field(ge=0)
    throughput: int = Field(ge=0)
    congestion_imbalance: int = Field(ge=0)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: TaskName
    step_count: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    current_phase: Phase
    time_in_phase: int = Field(ge=0)
    queues: dict[str, int]
    total_queue: int = Field(ge=0)
    total_wait_time: int = Field(ge=0)
    average_wait_time_per_lane: dict[str, float]
    max_wait_time_per_lane: dict[str, int]
    total_throughput: int = Field(ge=0)
    fairness_index: float = Field(ge=0.0, le=1.0)
    congestion_imbalance: int = Field(ge=0)
    regime_label: TrafficRegime
    switches_used: int = Field(ge=0)
    intersections: dict[IntersectionId, IntersectionObservation]
    network_dimensions: tuple[int, int]
    network_imbalance: float = Field(ge=0.0, le=1.0)
    last_action_error: Optional[str] = None
    done: bool


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Optional[ActionType] = None
    intersection_actions: Optional[dict[IntersectionId, ActionType]] = None


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0, le=1.0)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: Reward


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: TaskName
    description: str
    motivation: str
    objective: str
    stress_note: str
    max_steps: int = Field(gt=0)


class TaskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: TaskName
    steps_completed: int = Field(ge=0)
    total_reward: float = Field(ge=0.0)
    average_queue_length: float = Field(ge=0.0)
    average_wait_time: float = Field(ge=0.0)
    fairness_index: float = Field(ge=0.0, le=1.0)
    total_throughput: int = Field(ge=0)
    switches_used: int = Field(ge=0)
    final_total_queue: int = Field(ge=0)
    final_total_wait_time: int = Field(ge=0)
    network_imbalance: float = Field(ge=0.0, le=1.0)
    queue_volatility: float = Field(ge=0.0)
    peak_queue: int = Field(ge=0)
    gridlock_steps: int = Field(ge=0)
    moderate_congestion_steps: int = Field(ge=0)
    free_flow_steps: int = Field(ge=0)
    recovery_time: int = Field(ge=0)
    oscillation_count: int = Field(ge=0)
    sustained_fairness: float = Field(ge=0.0, le=1.0)
    task_metrics: dict[str, float] = Field(default_factory=dict)
    done: bool


class GradeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: TaskName
    score: float = Field(ge=0.0, le=1.0)
    average_queue_length: float = Field(ge=0.0)
    average_wait_time: float = Field(ge=0.0)
    fairness_index: float = Field(ge=0.0, le=1.0)
    total_throughput: int = Field(ge=0)
    switches_used: int = Field(ge=0)
    final_total_queue: int = Field(ge=0)
    network_imbalance: float = Field(ge=0.0, le=1.0)
    score_breakdown: dict[str, float]
    component_explanations: dict[str, str]
    overall_explanation: str
    details: str


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: Optional[TaskName] = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: Optional[str] = None
    intersection_actions: Optional[dict[str, str]] = None
