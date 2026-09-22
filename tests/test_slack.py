"""End-to-end tests for GET /plans/{id}/slack-report (read-only slack)."""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("PLANNER_DB_PATH", tmp.name)
    monkeypatch.setenv("PLANNER_SOLVER_TIME_LIMIT", "5")
    from importlib import reload

    from app import config, db, main

    reload(config)
    settings = config.load_settings()
    main.app.state.settings = settings
    db.init_db(settings.database_path)

    with TestClient(main.app) as c:
        yield c

    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


CHAIN_PAYLOAD = {
    # Stored waves: [[a], [b, c], [d]] (3 waves); the longest chain
    # a -> b/c -> d already uses all 3 waves, so every service is critical.
    "services": ["a", "b", "c", "d"],
    "dependencies": [
        {"service": "b", "depends_on": "a"},
        {"service": "c", "depends_on": "a"},
        {"service": "d", "depends_on": "b"},
        {"service": "d", "depends_on": "c"},
    ],
    "max_concurrency": 4,
}


def _create(client, payload):
    r = client.post("/plans", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def _report(client, plan_id):
    r = client.get(f"/plans/{plan_id}/slack-report")
    assert r.status_code == 200, r.text
    return r.json()


def _by_service(body):
    return {row["service"]: row for row in body["services"]}


# ---------------------------------------------------------------------------
# Slack semantics
# ---------------------------------------------------------------------------


def test_slack_binding_chain_every_service_is_critical(client):
    plan = _create(client, CHAIN_PAYLOAD)
    body = _report(client, plan["id"])

    assert body["plan_id"] == plan["id"]
    assert body["wave_count"] == 3
    assert body["max_concurrency"] == 4
    assert body["optimal"] is True
    assert body["critical_services"] == ["a", "b", "c", "d"]

    rows = _by_service(body)
    assert set(rows) == {"a", "b", "c", "d"}
    for service, current in (("a", 1), ("b", 2), ("c", 2), ("d", 3)):
        assert rows[service]["current_wave"] == current
        assert rows[service]["latest_wave"] == current
        assert rows[service]["slack"] == 0

    # Transitive critical downstream along the binding chains.
    assert rows["a"]["constrained_downstream"] == ["b", "c", "d"]
    assert rows["b"]["constrained_downstream"] == ["d"]
    assert rows["c"]["constrained_downstream"] == ["d"]
    assert rows["d"]["constrained_downstream"] == []


def test_slack_services_are_sorted_by_id(client):
    plan = _create(client, CHAIN_PAYLOAD)
    body = _report(client, plan["id"])
    assert [row["service"] for row in body["services"]] == ["a", "b", "c", "d"]
    assert body["critical_services"] == sorted(body["critical_services"])


def test_slack_capacity_headroom_gives_positive_slack(client):
    # a -> b plus three independent services, cap 3. The canonical minimum
    # uses 2 waves: [[a, c], [b, d, e]]. Moving 'a' later would force 'b'
    # into a third wave, so a and b are critical; 'c' can slide from wave 1
    # to wave 2 (the independent last-wave services d, e cannot move).
    payload = {
        "services": ["a", "b", "c", "d", "e"],
        "dependencies": [{"service": "b", "depends_on": "a"}],
        "max_concurrency": 3,
    }
    plan = _create(client, payload)
    assert [w["services"] for w in plan["waves"]] == [["a", "c"], ["b", "d", "e"]]

    body = _report(client, plan["id"])
    assert body["wave_count"] == 2
    assert body["critical_services"] == ["a", "b", "d", "e"]

    rows = _by_service(body)
    assert rows["c"]["current_wave"] == 1
    assert rows["c"]["latest_wave"] == 2
    assert rows["c"]["slack"] == 1
    assert rows["c"]["constrained_downstream"] == []

    # 'a' is pinned by its dependent 'b', which is itself critical.
    assert rows["a"]["slack"] == 0
    assert rows["a"]["constrained_downstream"] == ["b"]
    assert rows["b"]["slack"] == 0
    assert rows["b"]["constrained_downstream"] == []
    assert rows["d"]["slack"] == rows["e"]["slack"] == 0


def test_slack_independent_services_capacity_critical_last_wave(client):
    # Four independent services, cap 2 -> [[a, b], [c, d]]; wave 1 has room
    # for two more, so a and b each move to wave 2 and c, d are pinned by
    # the full last wave.
    plan = _create(
        client, {"services": ["a", "b", "c", "d"], "max_concurrency": 2}
    )
    assert [w["services"] for w in plan["waves"]] == [["a", "b"], ["c", "d"]]

    body = _report(client, plan["id"])
    rows = _by_service(body)
    assert rows["a"]["slack"] == rows["b"]["slack"] == 1
    assert rows["c"]["slack"] == rows["d"]["slack"] == 0
    assert body["critical_services"] == ["c", "d"]
    # No dependency edges: nothing is downstream of anything.
    assert all(row["constrained_downstream"] == [] for row in body["services"])


def test_slack_fault_domain_members_spread_with_slack(client):
    # Three independent services in one anti-affinity domain, cap 3: the
    # domain forces 3 waves [[a], [b], [c]]. 'c' is pinned at the last
    # wave; 'b' can move one wave later, 'a' two waves later.
    payload = {
        "services": ["a", "b", "c"],
        "dependencies": [],
        "max_concurrency": 3,
        "fault_domains": [{"domain": "az-1", "services": ["a", "b", "c"]}],
    }
    plan = _create(client, payload)
    assert [w["services"] for w in plan["waves"]] == [["a"], ["b"], ["c"]]

    body = _report(client, plan["id"])
    rows = _by_service(body)
    assert rows["a"]["slack"] == 2
    assert rows["a"]["latest_wave"] == 3
    assert rows["b"]["slack"] == 1
    assert rows["b"]["latest_wave"] == 3
    assert rows["c"]["slack"] == 0
    assert rows["c"]["latest_wave"] == 3
    assert body["critical_services"] == ["c"]


def test_slack_combined_dependencies_concurrency_fault_domain(client):
    # README payment-platform: [db] -> [auth] -> [api] -> [cache, web],
    # db/cache share a fault domain, cap 2, 4 waves. The dependency chain
    # db -> auth -> api -> web spans all 4 waves; cache is pinned at the
    # last wave by the fault domain (db must stay strictly before it).
    payload = {
        "name": "payment-platform",
        "services": ["web", "api", "auth", "db", "cache"],
        "dependencies": [
            {"service": "web", "depends_on": "api"},
            {"service": "api", "depends_on": "auth"},
            {"service": "api", "depends_on": "db"},
            {"service": "auth", "depends_on": "db"},
        ],
        "max_concurrency": 2,
        "fault_domains": [{"domain": "az-a", "services": ["db", "cache"]}],
    }
    plan = _create(client, payload)
    assert [w["services"] for w in plan["waves"]] == [
        ["db"],
        ["auth"],
        ["api"],
        ["cache", "web"],
    ]

    body = _report(client, plan["id"])
    assert body["wave_count"] == 4
    assert body["optimal"] is True
    assert body["critical_services"] == ["api", "auth", "cache", "db", "web"]

    rows = _by_service(body)
    for row in rows.values():
        assert row["slack"] == 0
    assert rows["db"]["constrained_downstream"] == ["api", "auth", "web"]
    assert rows["auth"]["constrained_downstream"] == ["api", "web"]
    assert rows["api"]["constrained_downstream"] == ["web"]
    # cache has no dependents; web is a leaf of the last wave.
    assert rows["cache"]["constrained_downstream"] == []
    assert rows["web"]["constrained_downstream"] == []


def test_slack_single_service(client):
    plan = _create(client, {"services": ["x"], "max_concurrency": 1})
    body = _report(client, plan["id"])
    assert body["wave_count"] == 1
    assert body["critical_services"] == ["x"]
    row = body["services"][0]
    assert row["current_wave"] == row["latest_wave"] == 1
    assert row["slack"] == 0
    assert row["constrained_downstream"] == []


# ---------------------------------------------------------------------------
# Error / read-only / determinism
# ---------------------------------------------------------------------------


def test_slack_missing_plan_is_404(client):
    r = client.get("/plans/9999/slack-report")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_found"
    issue = body["issues"][0]
    assert issue["code"] == "plan_not_found"
    assert issue["constraint"] == "plan_id"
    assert issue["context"] == {"id": 9999}


def test_slack_report_is_read_only(client):
    plan = _create(client, CHAIN_PAYLOAD)
    before = client.get("/plans").json()["total"]

    for _ in range(2):
        assert client.get(f"/plans/{plan['id']}/slack-report").status_code == 200

    assert client.get("/plans").json()["total"] == before
    fetched = client.get(f"/plans/{plan['id']}").json()
    assert fetched["waves"] == plan["waves"]
    assert fetched["wave_count"] == plan["wave_count"]


def test_slack_report_is_deterministic(client):
    plan = _create(client, CHAIN_PAYLOAD)
    responses = [
        client.get(f"/plans/{plan['id']}/slack-report").json() for _ in range(3)
    ]
    assert responses[0] == responses[1] == responses[2]


def test_slack_report_deterministic_under_capacity_headroom(client):
    payload = {
        "services": ["a", "b", "c", "d", "e"],
        "dependencies": [{"service": "b", "depends_on": "a"}],
        "max_concurrency": 3,
    }
    plan = _create(client, payload)
    responses = [
        client.get(f"/plans/{plan['id']}/slack-report").json() for _ in range(3)
    ]
    assert responses[0] == responses[1] == responses[2]
