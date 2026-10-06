import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from nirnaya.serve import MAX_BODY_BYTES, concentration, make_handler


@dataclass
class _Diagnostics:
    state_tokens: int = 10
    state_tokens_used: int = 10
    question_tokens: int = 5
    question_tokens_used: int = 5
    option_tokens: int = 4
    option_tokens_used: int = 4
    sequence_tokens: int = 25


class _FakeRuntime:
    def __init__(self, truncate: bool = False):
        self.truncate = truncate
        self.calls: list[Mapping[str, Mapping[str, object]]] = []

    def input_diagnostics(self, state, questions):
        return {
            name: _Diagnostics(state_tokens_used=3 if self.truncate else 10) for name in questions
        }

    def predict(self, state, questions):
        self.calls.append(questions)
        results = {}
        for name, question in questions.items():
            if question["type"] == "choice":
                keys = list(question["criteria"])
                values = [0.7, 0.2, 0.1][: len(keys)]
                results[name] = {"distribution": dict(zip(keys, values, strict=True))}
            elif question["type"] == "noul":
                results[name] = {"distribution": {"false": 0.25, "true": 0.75}}
            else:
                results[name] = {
                    "distribution": {"0": 0.5, "1": 0.3, "2": 0.2},
                    "expected_score": 0.7,
                }
        return results


@pytest.fixture
def serve() -> Iterator:
    servers: list[ThreadingHTTPServer] = []

    def start(runtime: _FakeRuntime, api_key: str | None = None) -> str:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(runtime, model_name="flash", api_key=api_key)
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/v1/systemone"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def _post(url: str, body: object, headers: Mapping[str, str] | None = None) -> tuple[int, dict]:
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    request = Request(
        url, data=data, headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urlopen(request) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


_QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Payments", "technical": None, "other": "Anything else"},
    },
    "refund": {"type": "noul", "instructions": "Refund requested?", "criteria": {"true": "Yes"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Low", "Mid", "High"]},
}


def test_system_one_response_shape_matches_the_jev_contract(serve) -> None:
    runtime = _FakeRuntime()
    status, body = _post(
        serve(runtime),
        {"model": "flash", "state": {"ticket": "charged twice"}, "questions": _QUESTIONS},
    )

    assert status == 200
    assert body["model"] == "flash"
    assert body["usage"] == {"input_tokens": 75, "output_tokens": 0}
    team = body["answers"]["team"]
    assert team["choice"] == "billing"
    assert team["probabilities"] == {"billing": 0.7, "technical": 0.2, "other": 0.1}
    assert team["confidence"] == pytest.approx(concentration([0.7, 0.2, 0.1]))
    assert body["answers"]["refund"] == {"type": "noul", "noul": 0.75}
    urgency = body["answers"]["urgency"]
    assert urgency["score"] == 0.7
    assert urgency["legend"] == {"0": "Low", "1": "Mid", "2": "High"}
    assert runtime.calls[0]["refund"]["criteria"] == {"false": None, "true": "Yes"}


def test_requests_outside_the_supported_contract_are_rejected(serve) -> None:
    url = serve(_FakeRuntime())
    eleven = {f"label{index}": None for index in range(11)}

    assert _post(url, {"state": "x", "questions": {}})[0] == 422
    assert (
        _post(url, {"state": "x", "questions": {"q": {"type": "rank", "instructions": "?"}}})[0]
        == 422
    )
    status, body = _post(
        url,
        {
            "state": "x",
            "questions": {"q": {"type": "choice", "instructions": "?", "criteria": eleven}},
        },
    )
    assert status == 422 and "2 to 10" in body["error"]
    assert _post(url, {"model": "nimble", "state": "x", "questions": _QUESTIONS})[0] == 404
    assert _post(url, b"{not json")[0] == 400
    assert _post(url, b" " * (MAX_BODY_BYTES + 1))[0] == 413


def test_truncated_prompts_are_rejected_instead_of_silently_cut(serve) -> None:
    status, body = _post(
        serve(_FakeRuntime(truncate=True)), {"state": "x", "questions": _QUESTIONS}
    )

    assert status == 400
    assert "exceeds the model context" in body["error"]


def test_bearer_key_is_required_when_configured(serve) -> None:
    url = serve(_FakeRuntime(), api_key="secret")
    body = {"state": "x", "questions": _QUESTIONS}

    assert _post(url, body)[0] == 401
    assert _post(url, body, {"Authorization": "Bearer wrong"})[0] == 401
    assert _post(url, body, {"Authorization": "Bearer secret"})[0] == 200


def test_concentration_is_zero_for_uniform_and_one_for_certain() -> None:
    assert concentration([0.25] * 4) == pytest.approx(0.0)
    assert concentration([1.0, 0.0, 0.0]) == pytest.approx(1.0)
