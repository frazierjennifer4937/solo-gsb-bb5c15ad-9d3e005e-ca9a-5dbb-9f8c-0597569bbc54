"""Read-only service-slack report for a stored minimum-wave plan.

For every service the report computes how many waves it can be moved
**later** while keeping the plan feasible: the total number of waves must
not grow, and every dependency (strictly earlier release), the per-wave
concurrency cap and fault-domain anti-affinity must still hold. This is the
classical scheduling notion of *total float*.

The latest feasible wave of each service is found with the **same
deterministic CP-SAT solver** used for planning (fixed random seed, a single
search worker): the wave model is rebuilt on the fixed horizon
``0 .. wave_count - 1`` (so the wave count cannot increase) with that
service's position maximized. The current stored assignment seeds every
solve, so identical inputs always yield identical slack values.

A service whose slack is zero cannot move at all: it is on a binding
constraint chain and is reported as a *critical service*. For each critical
service the report also lists its transitive critical **downstream**
services — services that (transitively) depend on it and are themselves
critical, i.e. the portion of the binding chain pinned behind it.

The stored plan is read but never modified: the report is computed on
demand and nothing is persisted.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .planner import service_latest_waves


@dataclass(frozen=True)
class ServiceSlack:
    service: str
    current_wave: int  # 1-based wave in the stored plan
    latest_wave: int  # latest feasible 1-based wave within wave_count
    slack: int  # latest_wave - current_wave
    constrained_downstream: list[str]  # transitive critical dependents


@dataclass(frozen=True)
class SlackReport:
    services: list[ServiceSlack]
    wave_count: int
    max_concurrency: int
    critical_services: list[str]
    optimal: bool


def _critical_downstream(
    services: list[str], edges: list[tuple[str, str]], critical: set[str]
) -> dict[str, list[str]]:
    """Transitive critical dependents of every service, sorted.

    Edges point ``service -> depends_on`` (prerequisite direction), so the
    reverse adjacency lists each service's direct dependents. Only
    dependents that are themselves critical are reported.
    """
    dependents: dict[str, list[str]] = {s: [] for s in services}
    for service, depends_on in edges:
        dependents[depends_on].append(service)

    result: dict[str, list[str]] = {}
    for s in services:
        seen: set[str] = set()
        stack = list(dependents[s])
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(dependents[node])
        result[s] = sorted(seen & critical)
    return result


def compute_slack(
    services: list[str],
    edges: list[tuple[str, str]],
    fault_domains: list[dict],
    concurrency: int,
    wave_members: list[list[str]],
    settings: Settings,
) -> SlackReport:
    """Compute the slack report for one stored plan.

    ``services`` lists every service of the plan, ``edges`` its dependency
    edges as ``(service, depends_on)``, ``fault_domains`` the stored
    ``[{"domain", "services"}]`` entries and ``wave_members`` the stored
    waves in order. Nothing is persisted by the caller.
    """
    ordered = sorted(services)
    wave_count = len(wave_members)
    current_pos = {
        service: index
        for index, members in enumerate(wave_members)
        for service in members
    }
    fault_cliques = [
        sorted(fd["services"])
        for fd in fault_domains
        if len(fd["services"]) >= 2
    ]

    latest_pos, proven = service_latest_waves(
        ordered,
        edges,
        fault_cliques,
        concurrency,
        wave_count,
        current_pos,
        settings,
    )

    slack_of = {s: latest_pos[s] - current_pos[s] for s in ordered}
    critical = {s for s in ordered if slack_of[s] == 0}
    downstream = _critical_downstream(ordered, edges, critical)

    rows = [
        ServiceSlack(
            service=s,
            current_wave=current_pos[s] + 1,
            latest_wave=latest_pos[s] + 1,
            slack=slack_of[s],
            constrained_downstream=downstream[s],
        )
        for s in ordered
    ]
    return SlackReport(
        services=rows,
        wave_count=wave_count,
        max_concurrency=concurrency,
        critical_services=sorted(critical),
        optimal=proven,
    )
