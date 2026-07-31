from typing import Any, Callable, Dict

from .models import RepairAction, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register(self, action_name: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        if action_name in self._handlers:
            raise ValueError(f"Tool handler already registered: {action_name}")
        self._handlers[action_name] = handler

    def execute(self, action: RepairAction) -> ToolResult:
        try:
            handler = self._handlers[action.name]
        except KeyError as exc:
            raise KeyError(f"No tool handler registered for: {action.name}") from exc
        return ToolResult(action, handler(action.payload))

    @property
    def action_names(self):
        return tuple(sorted(self._handlers))
