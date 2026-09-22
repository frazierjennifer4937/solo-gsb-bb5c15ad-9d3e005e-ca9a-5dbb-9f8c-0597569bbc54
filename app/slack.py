"""Service slack (wave-level float) report for a stored release plan.

Given the *stored* minimum-wave schedule, every service is assigned the
maximum wave it can move to **backwards** (later) while:

* the total number of waves stays exactly the stored minimum (no extra wave);
* every dependency edge still points at a strictly earlier wave;
* the stored per-wave concurrency cap and fault-domain anti-affinity keep
  holding.

That maximum is found with the same deterministic CP-SAT solver used for
planning: the identical wave model is reused with the horizon fixed to the
proven minimum wave count, and each service's position is maximized in turn
(one small solve per service, each hinted with the stored feasible
assignment). A service whose stored wave equals its latest feasible wave has
**zero slack** — it is critical; moving it any later would either add a wave
or break a constraint. For each critical service the report also lists the
downstream services constrained by it: critical services reachable over
dependency edges that are "tight" (exactly one wave apart in the stored
schedule), i.e. the zero-slack continuation of its critical chain(s).

The stored plan is read but never modified: the operation is read-only and
nothing is persisted. The solver uses the same fixed seed and single search
worker as planning, so identical plans yield identical reports.
"""
from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model

from .config import Settings
from .planner import FEASIBLE_STATUSES, _build_wave_model, _make_solver


@dataclass(frozen=True)
class ServiceSlack:
    service: str
    wave: int  # stored wave, 1-based
    latest_wave: int  # latest feasible wave, 1-based
    slack: int  # latest_wave - wave: maximum waves it can move later


@dataclass(frozen=True)
class SlackOutput:
    wave_count: int
    # True when every per-service maximum was proven optimal; False means
    # some slack values may be conservative lower bounds after a timeout.
    optimal: bool
    services: list[ServiceSlack]  # one row per service, sorted by service id
    critical: list[str]  # zero-slack services, sorted
    # critical service -> zero-slack downstream services reachable over tight
    # dependency edges, sorted (only entries for critical services)
    constrained_downstream: dict[str, list[str]]


def _latest_positions(
    services: list[str],
    edges: list[tuple[str, str]],
    fault_cliques: list[list[str]],
    wave_count: int,
    concurrency: int,
    stored_pos: dict[str, int],
    settings: Settings,
) -> tuple[dict[str, int], bool]:
    """Latest feasible wave (0-based) of every service within ``wave_count``.

    One solve per service: the standard wave model is constrained to the
    stored minimum horizon (``max_pos == wave_count - 1``) and the moved
    service's position is maximized. The stored assignment is passed as a
    hint so every solve starts from a known feasible solution. ``optimal``
    is the conjunction of the per-solve proofs; if a solve times out after
    only a feasible solution, its value is a conservative lower bound on
    the true maximum.
    """
    latest: dict[str, int] = {}
    optimal = True
    for target in services:
        model, pos, max_pos = _build_wave_model(
            services, edges, fault_cliques, wave_count, concurrency
        )
        # The horizon is exactly the stored minimum: no wave may be added.
        model.Add(max_pos == wave_count - 1)
        model.Maximize(pos[target])
        model.AddHint(max_pos, wave_count - 1)
        for s, w in stored_pos.items():
            model.AddHint(pos[s], w)
        solver = _make_solver(settings)
        status = solver.Solve(model)
        if status not in FEASIBLE_STATUSES:
            # Defensive: the stored assignment itself is a feasible solution
            # within the stored horizon.
            raise RuntimeError(
                "Unexpected solver infeasibility while computing service slack"
            )
        latest[target] = int(solver.Value(pos[target]))
        optimal = optimal and status == cp_model.OPTIMAL
    return latest, optimal


def compute_slack(
    services: list[str],
    edges: list[tuple[str, str]],
    domains: list[dict],
    concurrency: int,
    wave_members: list[list[str]],
    settings: Settings,
) -> SlackOutput:
    """Compute the read-only service-slack report for a stored plan.

    ``services`` lists every service of the stored plan, ``edges`` its
    dependency edges as ``(service, depends_on)``, ``domains`` the stored
    fault domains ``[{"domain", "services"}]`` and ``wave_members`` the
    stored waves in order (members sorted). Nothing is persisted.
    """
    wave_count = len(wave_members)
    stored_pos = {
        service: index
        for index, members in enumerate(wave_members)
        for service in members
    }
    fault_cliques = [
        sorted(fd["services"]) for fd in domains if len(fd["services"]) >= 2
    ]

    latest, optimal = _latest_positions(
        services,
        edges,
        fault_cliques,
        wave_count,
        concurrency,
        stored_pos,
        settings,
    )

    rows = [
        ServiceSlack(
            service=service,
            wave=stored_pos[service] + 1,
            latest_wave=latest[service] + 1,
            slack=latest[service] - stored_pos[service],
        )
        for service in sorted(services)
    ]
    critical = [row.service for row in rows if row.slack == 0]
    critical_set = set(critical)

    # Tight dependency successors of s: direct downstream services stored
    # exactly one wave later. A downstream is "constrained by" a critical
    # service when it is itself critical and reachable over a chain of such
    # tight edges — the zero-slack continuation of the critical chain.
    tight_successors: dict[str, list[str]] = {s: [] for s in services}
    for service, depends_on in edges:  # service is downstream of depends_on
        if stored_pos[service] == stored_pos[depends_on] + 1:
            tight_successors[depends_on].append(service)

    def constrained_downstream(start: str) -> list[str]:
        found: set[str] = set()
        stack = [start]
        seen: set[str] = {start}
        while stack:
            node = stack.pop()
            for nxt in tight_successors.get(node, []):
                if nxt in seen or nxt not in critical_set:
                    continue
                seen.add(nxt)
                found.add(nxt)
                stack.append(nxt)
        return sorted(found)

    return SlackOutput(
        wave_count=wave_count,
        optimal=optimal,
        services=rows,
        critical=critical,
        constrained_downstream={
            service: constrained_downstream(service) for service in critical
        },
    )
