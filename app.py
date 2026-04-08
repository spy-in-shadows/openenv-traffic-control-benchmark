from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse

from compare_baselines import run_policy
from traffic_env.env import TrafficSignalEnv
from traffic_env.explain import explain_transition
from traffic_env.models import ResetRequest, StepRequest
from traffic_env.tasks import TASK_DEFINITIONS

app = FastAPI(title="Traffic OpenEnv")
env = TrafficSignalEnv()
env.reset()
BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "traffic_openenv"}


@app.get("/demo")
def demo() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "demo.html")


@app.get("/tasks")
def tasks() -> dict[str, list[dict[str, str | int]]]:
    return {
        "tasks": [
            {
                "name": task.name,
                "description": task.description,
                "motivation": task.motivation,
                "objective": task.objective,
                "stress_note": task.stress_note,
                "max_steps": task.max_steps,
            }
            for task in TASK_DEFINITIONS.values()
        ]
    }


@app.get("/baselines")
def baselines() -> dict[str, list[dict[str, object]]]:
    return {
        "policies": [
            {"name": policy_name, "results": run_policy(policy_name)}
            for policy_name in ("always_keep", "alternating_switch", "queue_aware_heuristic")
        ]
    }


@app.post("/reset")
def reset(request: ResetRequest | None = Body(default=None)) -> dict[str, object]:
    task_name = request.task_name if request is not None else None
    observation = env.reset(task_name)
    return {"observation": observation.model_dump()}


@app.post("/step")
def step(request: StepRequest) -> dict[str, object]:
    previous_observation = env.state()
    action_payload: dict[str, object] = {}
    if request.action_type is not None:
        action_payload["action_type"] = request.action_type
    if request.intersection_actions is not None:
        action_payload["intersection_actions"] = request.intersection_actions
    result = env.step(action_payload if action_payload else {"action_type": "keep"})
    analysis = explain_transition(
        previous_observation,
        request.action_type or "network_action",
        result.reward.value,
        result.observation,
    )
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward.model_dump(),
        "analysis": analysis,
    }


@app.get("/state")
def state() -> dict[str, object]:
    summary = env.episode_summary()
    grade = env.grade()
    return {
        "observation": env.state().model_dump(),
        "summary": summary.model_dump(),
        "grade": grade.model_dump(),
    }
