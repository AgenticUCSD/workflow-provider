import json
from typing import TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from utils.config import OPENAI_API_KEY


model = ChatOpenAI(
    model="gpt-4.1",
    api_key=OPENAI_API_KEY,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


def extract_structured_output(
    result: object,
    model_type: type[ModelT],
    *,
    raise_on_error: bool = True,
) -> ModelT | None:
    if isinstance(result, model_type):
        return result

    def _validate(payload: object) -> ModelT | None:
        if payload is None:
            return None
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return None
        try:
            return model_type.model_validate(payload)
        except Exception:
            return None

    parsed = _validate(result)
    if parsed is not None:
        return parsed

    if isinstance(result, dict):
        for key in ("output", "structured_output"):
            parsed = _validate(result.get(key))
            if parsed is not None:
                return parsed

        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                parsed = _validate(getattr(message, "content", None))
                if parsed is not None:
                    return parsed

                additional = getattr(message, "additional_kwargs", None)
                if isinstance(additional, dict):
                    for key in ("tool_calls", "parsed", "structured_output", "output"):
                        parsed = _validate(additional.get(key))
                        if parsed is not None:
                            return parsed

                tool_calls = getattr(message, "tool_calls", None)
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
                        parsed = _validate(args)
                        if parsed is not None:
                            return parsed

    if raise_on_error:
        raise ValueError(f"Could not parse {model_type.__name__} from agent result")
    return None
