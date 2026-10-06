"""System One-compatible HTTP server (`POST /v1/systemone`) for Flash checkpoints."""

import argparse
import hmac
import json
import logging
import math
import os
import threading
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

from nirnaya.data.schemas import serialize_state

MAX_BODY_BYTES = 64 * 1024
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 10
MAX_SCORE_LEVELS = 26
DEFAULT_MODEL_NAME = "nirnaya-flash"

_LOGGER = logging.getLogger("nirnaya.serve")


class DecisionRuntime(Protocol):
    def predict(
        self, state: Any, questions: Mapping[str, Mapping[str, object]]
    ) -> Mapping[str, Mapping[str, object]]: ...

    def input_diagnostics(
        self, state: Any, questions: Mapping[str, Mapping[str, object]]
    ) -> Mapping[str, Any]: ...


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status


def _invalid(message: str) -> RequestError:
    return RequestError(HTTPStatus.UNPROCESSABLE_ENTITY, message)


def _is_content(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, (Mapping, list))


def translate_questions(raw: object) -> dict[str, dict[str, object]]:
    """Validate System One questions and translate them to the runtime request format."""
    if not isinstance(raw, Mapping) or not 1 <= len(raw) <= MAX_QUESTIONS:
        raise _invalid(f"questions must be an object with 1 to {MAX_QUESTIONS} entries")
    translated: dict[str, dict[str, object]] = {}
    for name, question in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise _invalid("question names must not be blank")
        if not isinstance(question, Mapping):
            raise _invalid(f"question {name!r} must be an object")
        instructions = question.get("instructions")
        if not _is_content(instructions):
            raise _invalid(f"question {name!r} requires non-empty instructions")
        text = serialize_state(instructions)  # type: ignore[arg-type]
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
                raise _invalid(f"choice question {name!r} needs 2 to {MAX_CHOICE_OPTIONS} criteria")
            if any(not isinstance(key, str) or not key.strip() for key in criteria) or any(
                value is not None and not isinstance(value, str) for value in criteria.values()
            ):
                raise _invalid(f"choice question {name!r} criteria must map names to text or null")
            translated[name] = {"type": "choice", "instructions": text, "criteria": dict(criteria)}
        elif kind == "noul":
            criteria = {} if criteria is None else criteria
            if (
                not isinstance(criteria, Mapping)
                or set(criteria) - {"false", "true"}
                or any(not isinstance(value, str) for value in criteria.values())
            ):
                raise _invalid(f"noul question {name!r} criteria may only describe false and true")
            entry: dict[str, object] = {"type": "noul", "instructions": text}
            if criteria:
                entry["criteria"] = {"false": criteria.get("false"), "true": criteria.get("true")}
            translated[name] = entry
        elif kind == "score":
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= MAX_SCORE_LEVELS
                or any(not isinstance(item, str) for item in criteria)
            ):
                raise _invalid(
                    f"score question {name!r} needs 2 to {MAX_SCORE_LEVELS} text criteria"
                )
            translated[name] = {"type": "score", "instructions": text, "criteria": list(criteria)}
        else:
            raise _invalid(f"question {name!r} type must be choice, noul, or score")
    return translated


def concentration(probabilities: Sequence[float]) -> float:
    """Return 1 - H(p) / ln(N), the System One confidence definition."""
    if len(probabilities) < 2:
        return 1.0
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities))))


def system_one_answer(
    question: Mapping[str, object], result: Mapping[str, object]
) -> dict[str, object]:
    """Convert one runtime result into the System One answer shape."""
    probabilities = {str(key): float(value) for key, value in result["distribution"].items()}  # type: ignore[union-attr]
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probabilities["true"]}
    answer: dict[str, object] = {"type": kind}
    if kind == "choice":
        # max() keeps the first maximum, so ties select the first option in request order.
        answer["choice"] = max(probabilities, key=probabilities.__getitem__)
    else:
        answer["score"] = float(result["expected_score"])  # type: ignore[arg-type]
        criteria = question["criteria"]
        answer["legend"] = {str(index): text for index, text in enumerate(criteria)}  # type: ignore[arg-type]
    answer["probabilities"] = probabilities
    answer["confidence"] = concentration(list(probabilities.values()))
    return answer


def handle_system_one(
    runtime: DecisionRuntime,
    body: object,
    *,
    model_name: str,
    lock: threading.Lock,
) -> dict[str, object]:
    """Validate one request, score every question, and build the System One response."""
    if not isinstance(body, Mapping):
        raise _invalid("request body must be a JSON object")
    requested_model = body.get("model")
    if requested_model is not None and requested_model != model_name:
        raise RequestError(HTTPStatus.NOT_FOUND, f"model {requested_model!r} is not served here")
    state = body.get("state")
    if not _is_content(state):
        raise _invalid("state must be a non-empty string, object, or array")
    questions = translate_questions(body.get("questions"))
    try:
        with lock:
            diagnostics = runtime.input_diagnostics(state, questions)
            for name, item in diagnostics.items():
                if (
                    item.state_tokens_used < item.state_tokens
                    or item.question_tokens_used < item.question_tokens
                    or item.option_tokens_used < item.option_tokens
                ):
                    raise RequestError(
                        HTTPStatus.BAD_REQUEST,
                        f"rendered prompt for question {name!r} exceeds the model context",
                    )
            results = runtime.predict(state, questions)
    except (TypeError, ValueError) as error:
        raise _invalid(str(error)) from error
    return {
        "model": model_name,
        "answers": {
            name: system_one_answer(question, results[name]) for name, question in questions.items()
        },
        "usage": {
            "input_tokens": sum(item.sequence_tokens for item in diagnostics.values()),
            "output_tokens": 0,
        },
    }


def make_handler(
    runtime: DecisionRuntime,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one loaded runtime."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "nirnaya-serve"

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(HTTPStatus.OK, {"status": "ok", "model": model_name})
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self._authorized():
                self._send(HTTPStatus.UNAUTHORIZED, {"error": "missing or invalid bearer token"})
                return
            try:
                body = self._read_json()
                response = handle_system_one(runtime, body, model_name=model_name, lock=lock)
            except RequestError as error:
                self._send(error.status, {"error": str(error)})
                return
            except Exception:
                _LOGGER.exception("scoring failed")
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "scoring failed"})
                return
            self._send(HTTPStatus.OK, response)

        def _authorized(self) -> bool:
            if api_key is None:
                return True
            header = self.headers.get("Authorization", "")
            scheme, _, token = header.partition(" ")
            return scheme.lower() == "bearer" and hmac.compare_digest(
                token.strip().encode(), api_key.encode()
            )

        def _read_json(self) -> object:
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError as error:
                raise RequestError(
                    HTTPStatus.LENGTH_REQUIRED, "Content-Length is required"
                ) from error
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                raise RequestError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body must not exceed 64 KiB"
                )
            try:
                return json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RequestError(HTTPStatus.BAD_REQUEST, "request body must be JSON") from error

        def _send(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def load_runtime(
    model: str, backend: str | None = None, device: str | None = None
) -> DecisionRuntime:
    """Load a release directory or Hugging Face repository with MLX or PyTorch."""
    if backend is None:
        try:
            import mlx.core  # noqa: F401

            backend = "mlx"
        except ImportError:
            backend = "torch"
    if backend == "torch":
        from nirnaya.hf.runtime import TorchFlashRuntime

        return TorchFlashRuntime.from_pretrained(model, device=device)
    from nirnaya.inference.mlx_runtime import MLXFlashRuntime

    return MLXFlashRuntime.from_pretrained(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve typed decisions on POST /v1/systemone")
    parser.add_argument("model", help="release directory or Hugging Face repository id")
    parser.add_argument("--backend", choices=("mlx", "torch"))
    parser.add_argument("--device", help="PyTorch device, such as cpu, cuda, or mps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    arguments = parser.parse_args()

    runtime = load_runtime(arguments.model, arguments.backend, arguments.device)
    api_key = os.environ.get("NIRNAYA_API_KEY") or None
    if api_key is None and arguments.host not in {"127.0.0.1", "localhost", "::1"}:
        _LOGGER.warning(
            "serving on %s without NIRNAYA_API_KEY; requests are unauthenticated", arguments.host
        )
    handler = make_handler(runtime, model_name=arguments.model_name, api_key=api_key)
    server = ThreadingHTTPServer((arguments.host, arguments.port), handler)
    print(
        f"serving {arguments.model_name} on http://{arguments.host}:{arguments.port}/v1/systemone"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
