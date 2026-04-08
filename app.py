from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from compare_baselines import run_policy
from traffic_env.env import TrafficSignalEnv
from traffic_env.explain import explain_transition
from traffic_env.models import StepRequest
from traffic_env.tasks import TASK_DEFINITIONS

app = FastAPI(title="Traffic OpenEnv")
env = TrafficSignalEnv()
env.reset()
BASE_DIR = Path(__file__).resolve().parent


def run() -> None:
    import os

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host=host, port=port)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "traffic_openenv"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/demo")
def demo() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "demo.html")


@app.get("/web")
def web() -> FileResponse:
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
async def reset(request: Request) -> dict[str, object]:
    task_name = None
    try:
        payload = await request.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        raw_task_name = payload.get("task_name")
        if isinstance(raw_task_name, str) and raw_task_name in TASK_DEFINITIONS:
            task_name = raw_task_name
    observation = env.reset(task_name)
    observation_payload = observation.model_dump()
    return {"observation": observation_payload, **observation_payload}


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
    observation_payload = result.observation.model_dump()
    return {
        "observation": observation_payload,
        **observation_payload,
        "reward": result.reward.model_dump(),
        "analysis": analysis,
    }


@app.get("/state")
def state() -> dict[str, object]:
    observation = env.state()
    summary = env.episode_summary()
    grade = env.grade()
    observation_payload = observation.model_dump()
    return {
        "observation": observation_payload,
        **observation_payload,
        "summary": summary.model_dump(),
        "grade": grade.model_dump(),
    }
