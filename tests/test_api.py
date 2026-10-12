from __future__ import annotations

from conftest import gen
from fastapi.testclient import TestClient

from pharmasql.api import app, get_agent_factory


def client_with(agent) -> TestClient:
    app.dependency_overrides[get_agent_factory] = lambda: (lambda _config: agent)
    return TestClient(app)


def test_ask_returns_sql_rows_and_trace(make_agent):
    agent, _ = make_agent([gen("SELECT region_name FROM dim_region ORDER BY region_name")])
    resp = client_with(agent).post("/ask", json={"question": "List regions"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["columns"] == ["region_name"]
    assert len(body["rows"]) == 5
    assert body["sql"].startswith("SELECT")
    assert {t["step"] for t in body["trace"]} >= {"generate_sql", "validate", "execute"}


def test_abstention_has_no_sql(make_agent):
    agent, _ = make_agent([gen("", answerable=False)])
    body = client_with(agent).post("/ask", json={"question": "TV spend?"}).json()
    assert body["status"] == "abstained" and body["sql"] is None and body["rows"] == []


def test_rejects_bad_requests(make_agent):
    agent, _ = make_agent([gen("SELECT 1")])
    client = client_with(agent)
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "hi there", "config": "nope"}).status_code == 422
