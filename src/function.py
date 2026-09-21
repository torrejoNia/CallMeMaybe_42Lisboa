from pydantic import BaseModel, PrivateAttr
import json
from typing import Any

from src.encoder import Encoder


class Function(BaseModel):
    _name: str = PrivateAttr()
    _t_name: list[int] = PrivateAttr()
    _description: str = PrivateAttr()
    _t_description: list[int] = PrivateAttr()
    _params: dict[str, str] = PrivateAttr()
    _t_params: dict[str, list[int]] = PrivateAttr()
    _enums: dict[str, list[Any]] = PrivateAttr()
    _t_definition: list[int] = PrivateAttr()

    def __init__(self,
                 function: dict[str, Any],
                 encoder: Encoder):
        super().__init__()
        self._name: str = function['name']
        self._t_name = encoder.encode(self._name)
        self._description: str = function.get('description', '')
        self._t_description = encoder.encode(self._description)
        parameters = function.get('parameters', {})
        self._params = {
            k: v.get('type', 'string')
            for k, v in parameters.items()
        }
        self._t_params = {
            k: encoder.encode(v.get('type', 'string'))
            for k, v in parameters.items()
        }
        # Parameters restricted to a closed set of values, e.g.
        # {"firmware": {"type": "string", "enum": ["stable", "beta"]}}
        self._enums = {
            k: list(v['enum'])
            for k, v in parameters.items()
            if isinstance(v.get('enum'), list) and v['enum']
        }
        self._t_definition = encoder.encode(self._to_tool_schema())

    def _to_tool_schema(self) -> str:
        properties: dict[str, Any] = {}
        for name, arg_type in self._params.items():
            properties[name] = {"type": arg_type}
            if name in self._enums:
                properties[name]["enum"] = self._enums[name]
        return json.dumps({
            "name": self._name,
            "description": self._description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(self._params.keys())
            }
        })

    @property
    def t_definition(self) -> list[int]:
        return self._t_definition

    @property
    def name(self) -> str:
        return self._name

    @property
    def t_name(self) -> list[int]:
        return self._t_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def t_description(self) -> list[int]:
        return self._t_description

    @property
    def params(self) -> dict[str, str]:
        return self._params

    @property
    def t_params(self) -> dict[str, list[int]]:
        return self._t_params

    @property
    def enums(self) -> dict[str, list[Any]]:
        return self._enums

    @property
    def param_names(self) -> list[str]:
        return list(self._params.keys())
