from __future__ import annotations

from traffic_env.baselines import BASELINE_POLICIES
from traffic_env.env import TrafficSignalEnv
from traffic_env.tasks import get_task_names
from traffic_env.utils import strict_open_score


def run_policy(policy_name: str) -> list[dict[str, float | int | str]]:
    env = TrafficSignalEnv()
    policy = BASELINE_POLICIES[policy_name]
    results: list[dict[str, float | int | str]] = []

    for task_name in get_task_names():
        observation = env.reset(task_name)
        while not observation.done:
            policy_action = policy(observation)
            observation = env.step(policy_action).observation

        summary = env.episode_summary()
        grade = env.grade()
        results.append(
            {
                "task": task_name,
                "score": strict_open_score(grade.score),
                "avg_queue": round(summary.average_queue_length, 2),
                "avg_wait": round(summary.average_wait_time, 2),
                "fairness": round(summary.fairness_index, 3),
                "throughput": summary.total_throughput,
                "switches": summary.switches_used,
                "final_queue": summary.final_total_queue,
                "network_imbalance": round(summary.network_imbalance, 3),
            }
        )
    return results


def main() -> None:
    for policy_name in BASELINE_POLICIES:
        print(f"[POLICY] name={policy_name}")
        for result in run_policy(policy_name):
            print(
                "[RESULT] "
                f"task={result['task']} "
                f"score={result['score']:.3f} "
                f"avg_queue={result['avg_queue']:.2f} "
                f"avg_wait={result['avg_wait']:.2f} "
                f"fairness={result['fairness']:.3f} "
                f"throughput={result['throughput']} "
                f"switches={result['switches']} "
                f"network_imbalance={result['network_imbalance']:.3f} "
                f"final_queue={result['final_queue']}"
            )


if __name__ == "__main__":
    main()
