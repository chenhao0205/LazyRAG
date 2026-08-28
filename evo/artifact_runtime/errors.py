from collections.abc import Callable, Iterable, Sequence
from math import isfinite
from typing import TypeVar


_T = TypeVar('_T')


class ArtifactRuntimeError(Exception):
    """Base error for the artifact runtime."""


class DefinitionError(ArtifactRuntimeError, ValueError):
    """Raised when artifact or operation declarations are invalid."""


class PlanningError(ArtifactRuntimeError, RuntimeError):
    """Raised when an artifact snapshot cannot be planned."""


class OperationExecutionError(ArtifactRuntimeError, RuntimeError):
    """Raised when an operation execution unit fails."""


class OperationTimeoutError(OperationExecutionError, TimeoutError):
    """Raised when an operation exceeds its declared execution timeout."""


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f'{name} must be str')
    return value


def _text(value: object, name: str) -> str:
    value = _string(value, name)
    if not value.strip():
        raise DefinitionError(f'{name} must be non-empty')
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f'{name} must be int')
    if value < minimum:
        raise DefinitionError(f'{name} must be >= {minimum}')


def _positive_number(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f'{name} must be a number')
    if not isfinite(value) or value <= 0:
        raise DefinitionError(f'{name} must be finite and positive')


def _tuple_of(values: Iterable[object], expected_type: type[_T], message: str, *, nonempty: bool = False
              ) -> tuple[_T, ...]:
    items = tuple(values)
    if (nonempty and not items) or not all(isinstance(item, expected_type) for item in items):
        raise TypeError(message)
    return items


def _unique(values: Sequence[_T], message: str, key: Callable[[_T], object] | None = None) -> None:
    identities = values if key is None else map(key, values)
    if len(set(identities)) != len(values):
        raise DefinitionError(message)


def _known(value: object, allowed: Iterable[object], name: str) -> None:
    if value not in allowed:
        raise DefinitionError(f'unknown {name}: {value}')


def _number(value: object, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        suffix = ' or None' if optional else ''
        raise TypeError(f'{name} must be a number{suffix}')


def _as_exception(error: BaseException) -> Exception:
    if isinstance(error, Exception):
        return error
    return RuntimeError(f'{type(error).__name__}: {error}')


__all__ = [
    'ArtifactRuntimeError', 'DefinitionError', 'OperationExecutionError',
    'OperationTimeoutError', 'PlanningError', '_as_exception', '_integer', '_known', '_number',
    '_positive_number', '_string', '_text', '_tuple_of', '_unique',
]
