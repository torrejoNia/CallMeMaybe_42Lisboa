from pydantic import BaseModel, Field, field_validator
import json
from typing import Any

from src.encoder import Encoder


def _to_tool_schema(name: str,
                    description: str,
                    params: dict[str, str],
                    enums: dict[str, list[Any]]) -> str:
    """Renders the function as the JSON tool schema shown to the model."""

    properties: dict[str, Any] = {}
    for arg_name, arg_type in params.items():
        properties[arg_name] = {"type": arg_type}
        if arg_name in enums:
            properties[arg_name]["enum"] = enums[arg_name]
    return json.dumps({
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(params.keys())
        }
    })


class Function(BaseModel):
    """One callable function, as declared in functions_definition.json.

    The fields are declared rather than private, so pydantic validates the
    whole object when it is built from the input file: a name that is not a
    string, a parameter map that is not a mapping of strings, or a token list
    holding something other than integers is rejected at construction, before
    any prompt is processed.
    """

    name: str
    t_name: list[int]
    description: str = ''
    t_description: list[int] = Field(default_factory=list)
    params: dict[str, str] = Field(default_factory=dict)
    t_params: dict[str, list[int]] = Field(default_factory=dict)
    enums: dict[str, list[Any]] = Field(default_factory=dict)
    t_definition: list[int] = Field(default_factory=list)

    @field_validator('name')
    @classmethod
    def _check_name(cls, value: str) -> str:
        """Rejects a name that could not be written as a JSON string."""

        value = value.strip()
        if not value:
            raise ValueError('a function name must not be empty')
        if any(c in value for c in '"\\'):
            raise ValueError(f'unsupported characters in name: {value!r}')
        return value

    def __init__(self,
                 function: dict[str, Any],
                 encoder: Encoder):
        name = function['name']
        description = function.get('description', '')
        parameters = function.get('parameters', {})
        params = {
            k: v.get('type', 'string')
            for k, v in parameters.items()
        }
        # Parameters restricted to a closed set of values, e.g.
        # {"firmware": {"type": "string", "enum": ["stable", "beta"]}}
        enums = {
            k: list(v['enum'])
            for k, v in parameters.items()
            if isinstance(v.get('enum'), list) and v['enum']
        }
        schema = _to_tool_schema(name, description, params, enums)
        super().__init__(
            name=name,
            t_name=encoder.encode(name),
            description=description,
            t_description=encoder.encode(description),
            params=params,
            t_params={
                arg_name: encoder.encode(arg_type)
                for arg_name, arg_type in params.items()
            },
            enums=enums,
            t_definition=encoder.encode(schema),
        )

    @property
    def param_names(self) -> list[str]:
        return list(self.params.keys())
