from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import pickle
import signal
import socket
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import psutil

from .errors import DefinitionError, OperationExecutionError
from .operation import Operation, OperationContext, OperationInvocation, OperationResult
from .state import OperationEvent


_EVENT_LIMIT = 64 * 1024
_WORKER_ENTRYPOINT = 'from evo.artifact_runtime.execution import _main; _main()'
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _IsolatedRequest:
    module: str
    qualname: str
    run_id: str
    invocation_id: str
    partition_key: str
    inputs: tuple[tuple[str, object], ...]
    cleanup_timeout: float


class ExecutionHandle(Protocol):
    async def wait(self) -> OperationResult:
        ...

    async def terminate(self) -> None:
        ...


class ExecutionCleanupError(OperationExecutionError):
    def __init__(self, message: str, alive_processes: Sequence[psutil.Process] = (), unverified: bool = False) -> None:
        super().__init__(message, tuple(alive_processes), unverified)
        self.alive_processes = tuple(alive_processes)
        self.unverified = unverified

    def __str__(self) -> str:
        return str(self.args[0])

    @property
    def cleanup_pending(self) -> bool:
        return self.unverified or bool(self.alive_processes)


class _CooperativeHandle:
    def __init__(self, operation: Operation, task: asyncio.Task[OperationResult]) -> None:
        self._operation = operation
        self._task = task

    async def wait(self) -> OperationResult:
        return _validated_result(self._operation, await asyncio.shield(self._task))

    async def terminate(self) -> None:
        self._task.cancel()
        try:
            await asyncio.shield(self._task)
        except asyncio.CancelledError:
            if not self._task.cancelled():
                raise
        except Exception:
            pass


class _IsolatedHandle:
    def __init__(self, operation: Operation, process: asyncio.subprocess.Process, stdout_task: asyncio.Task[bytes],
                 stderr_task: asyncio.Task[bytes], event_task: asyncio.Task[None], result_path: Path,
                 directory: tempfile.TemporaryDirectory[str], terminate_timeout: float) -> None:
        self._operation = operation
        self._process = process
        self._stdout_task = stdout_task
        self._stderr_task = stderr_task
        self._event_task = event_task
        self._result_path = result_path
        self._directory = directory
        self._cleanup_timeout = terminate_timeout
        self._terminate_requested = False
        self._terminate_lock = asyncio.Lock()
        self._tracked_descendants: tuple[psutil.Process, ...] = ()
        self._unverified_cleanup: ExecutionCleanupError | None = None
        self._completion = asyncio.create_task(
            self._complete(),
            name=f'isolated:{operation.spec.op_id}:{process.pid}',
        )

    async def wait(self) -> OperationResult:
        result = await asyncio.shield(self._completion)
        if result is None:
            raise asyncio.CancelledError
        return result

    async def terminate(self) -> None:
        async with self._terminate_lock:
            if self._unverified_cleanup is not None:
                raise self._unverified_cleanup
            if self._completion.done() and not self._tracked_descendants:
                await _consume_completion(self._completion)
                return
            self._terminate_requested = True
            await self._terminate_tree()
            self._event_task.cancel()
            await _consume_completion(self._completion)

    async def _complete(self) -> OperationResult | None:
        process_waiter = asyncio.create_task(
            self._process.wait(),
            name=f'worker-wait:{self._process.pid}',
        )
        try:
            await asyncio.shield(process_waiter)
            if not self._terminate_requested and not self._result_path.is_file():
                _signal_process_group(self._process.pid, signal.SIGKILL)
            await self._finish_events()
            stdout, stderr = await self._output()
            return self._result(stdout, stderr)
        finally:
            if not process_waiter.done():
                process_waiter.cancel()
                await asyncio.gather(process_waiter, return_exceptions=True)
            self._directory.cleanup()

    async def _terminate_tree(self) -> None:
        try:
            self._tracked_descendants = await _terminate_process_tree(
                self._process,
                self._tracked_descendants,
                self._cleanup_timeout,
            )
        except ExecutionCleanupError as exc:
            self._tracked_descendants = exc.alive_processes
            if exc.unverified and not exc.alive_processes:
                self._unverified_cleanup = exc
            raise
        except Exception as exc:
            if isinstance(exc, OperationExecutionError):
                raise
            raise ExecutionCleanupError(str(exc) or type(exc).__name__) from exc
        self._unverified_cleanup = None

    async def _finish_events(self) -> None:
        if self._event_task.done():
            if self._event_task.cancelled():
                return
            error = self._event_task.exception()
            if error is not None:
                _LOG.warning('isolated operation event stream failed: %s', error)
            return
        try:
            async with asyncio.timeout(self._cleanup_timeout):
                await asyncio.shield(self._event_task)
        except TimeoutError:
            self._event_task.cancel()
            await asyncio.gather(self._event_task, return_exceptions=True)
            _LOG.warning('%s worker descendants kept its event pipe open', self._operation.spec.op_id)
        except asyncio.CancelledError:
            if not self._terminate_requested:
                raise
        except Exception as exc:
            _LOG.warning('isolated operation event stream failed: %s', exc)

    def _result(self, stdout: bytes, stderr: bytes) -> OperationResult | None:
        if self._terminate_requested:
            return None
        if self._result_path.is_file():
            try:
                response = pickle.loads(self._result_path.read_bytes())
                if not isinstance(response, dict):
                    raise TypeError('response must be a result mapping')
                return _validated_result(self._operation, OperationResult(response))
            except OperationExecutionError:
                raise
            except Exception as exc:
                raise OperationExecutionError(
                    f'{self._operation.spec.op_id} worker returned an invalid response'
                ) from exc
        detail = stderr.decode(errors='replace').strip() or stdout.decode(errors='replace').strip()
        if detail:
            raise OperationExecutionError(
                f'{self._operation.spec.op_id} worker failed: {detail}'
            )
        raise OperationExecutionError(
            f'{self._operation.spec.op_id} worker produced no result'
        )

    async def _output(self) -> tuple[bytes, bytes]:
        try:
            async with asyncio.timeout(self._cleanup_timeout):
                return await asyncio.gather(
                    self._stdout_task,
                    self._stderr_task,
                )
        except TimeoutError as exc:
            await asyncio.gather(
                self._stdout_task,
                self._stderr_task,
                return_exceptions=True,
            )
            raise OperationExecutionError(
                f'{self._operation.spec.op_id} worker descendants kept output pipes open'
            ) from exc


async def start_execution(invocation: OperationInvocation, ctx: OperationContext, inputs: Mapping[str, object], *,
                          terminate_timeout: float = 1.0) -> ExecutionHandle:
    if terminate_timeout <= 0:
        raise ValueError('terminate_timeout must be positive')
    if invocation.operation.spec.execution == 'cooperative':
        task = asyncio.create_task(
            invocation.operation(ctx, **dict(inputs)),
            name=f'cooperative:{invocation.invocation_id}',
        )
        return _CooperativeHandle(invocation.operation, task)
    if os.name != 'posix':
        raise OperationExecutionError(
            'isolated execution requires POSIX process sessions'
        )
    return await _start_isolated(invocation, ctx, inputs, terminate_timeout)


async def _start_isolated(invocation: OperationInvocation, ctx: OperationContext, inputs: Mapping[str, object],
                          terminate_timeout: float) -> _IsolatedHandle:
    directory = tempfile.TemporaryDirectory(prefix='artifact-operation-')
    event_reader: socket.socket | None = None
    event_writer: socket.socket | None = None
    process: asyncio.subprocess.Process | None = None
    stdout_task: asyncio.Task[bytes] | None = None
    stderr_task: asyncio.Task[bytes] | None = None
    event_task: asyncio.Task[None] | None = None
    try:
        root = Path(directory.name)
        request_path = root / 'request.pkl'
        result_path = root / 'result.pkl'
        request = _IsolatedRequest(
            invocation.operation.__module__,
            invocation.operation.__qualname__,
            ctx.run_id,
            ctx.invocation_id,
            ctx.partition_key,
            tuple(inputs.items()),
            terminate_timeout,
        )
        request_path.write_bytes(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))

        event_reader, event_writer = socket.socketpair()
        event_reader.setblocking(False)
        event_fd = event_writer.fileno()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            '-c',
            _WORKER_ENTRYPOINT,
            str(request_path),
            str(result_path),
            str(event_fd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            pass_fds=(event_fd,),
        )
        event_writer.close()
        event_writer = None
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_task = asyncio.create_task(_drain_output(process.stdout))
        stderr_task = asyncio.create_task(_drain_output(process.stderr))
        event_task = asyncio.create_task(_forward_operation_events(event_reader, ctx))
        handle = _IsolatedHandle(
            invocation.operation,
            process,
            stdout_task,
            stderr_task,
            event_task,
            result_path,
            directory,
            terminate_timeout,
        )
        event_reader = None
        return handle
    except BaseException as start_error:
        if event_reader is not None:
            event_reader.close()
        if event_writer is not None:
            event_writer.close()
        cleanup_errors: list[BaseException] = []
        try:
            if process is not None:
                try:
                    await _terminate_process_tree(process, (), terminate_timeout)
                except Exception as exc:
                    cleanup_errors.append(exc)
                    _signal_process_group(process.pid, signal.SIGKILL)
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    except Exception as fallback_signal_error:
                        cleanup_errors.append(fallback_signal_error)
                    try:
                        async with asyncio.timeout(terminate_timeout):
                            await asyncio.shield(process.wait())
                    except (asyncio.CancelledError, Exception) as fallback_error:
                        cleanup_errors.append(fallback_error)
            tasks = tuple(
                task
                for task in (stdout_task, stderr_task, event_task)
                if task is not None
            )
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            directory.cleanup()
        if cleanup_errors:
            detail = '; '.join(str(error) or type(error).__name__ for error in cleanup_errors)
            start_error.add_note(f'isolated worker cleanup also failed: {detail}')
        raise


async def _terminate_process_tree(process: asyncio.subprocess.Process, tracked: Sequence[psutil.Process], timeout: float
                                  ) -> tuple[psutil.Process, ...]:
    errors: list[Exception] = []
    try:
        discovered = await asyncio.to_thread(_descendants, process.pid)
    except Exception as exc:
        discovered = ()
        errors.append(exc)
    descendants = _merge_processes(tracked, discovered)
    errors.extend(await asyncio.to_thread(_signal_processes, descendants, signal.SIGTERM))
    group_error = _signal_process_group(process.pid, signal.SIGTERM)
    if group_error is not None:
        errors.append(group_error)

    root_exited, alive = await asyncio.gather(
        _wait_direct_process(process, timeout),
        asyncio.to_thread(_wait_processes, descendants, timeout),
    )
    if root_exited and not alive:
        if errors:
            raise ExecutionCleanupError(
                _termination_error_detail(errors),
                (),
                unverified=True,
            )
        return ()

    if not root_exited:
        try:
            discovered = await asyncio.to_thread(_descendants, process.pid)
        except Exception as exc:
            discovered = ()
            errors.append(exc)
        descendants = _merge_processes(alive, discovered)
    else:
        descendants = tuple(alive)
    errors.extend(await asyncio.to_thread(_signal_processes, descendants, signal.SIGKILL))
    group_error = _signal_process_group(process.pid, signal.SIGKILL)
    if group_error is not None:
        errors.append(group_error)

    root_exited, alive = await asyncio.gather(
        _wait_direct_process(process, timeout),
        asyncio.to_thread(_wait_processes, descendants, timeout),
    )
    if not root_exited or alive or errors:
        detail = []
        if not root_exited:
            detail.append(f'worker {process.pid}')
        if alive:
            detail.append('descendants ' + ','.join(str(item.pid) for item in alive))
        if errors:
            detail.append(_termination_error_detail(errors))
        raise ExecutionCleanupError(
            'isolated process tree cleanup was not verified: ' + '; '.join(detail),
            alive,
            unverified=bool(errors),
        )
    return ()


async def _wait_direct_process(process: asyncio.subprocess.Process, timeout: float) -> bool:
    try:
        async with asyncio.timeout(timeout):
            await asyncio.shield(process.wait())
        return True
    except TimeoutError:
        return False


def _descendants(pid: int) -> tuple[psutil.Process, ...]:
    try:
        return tuple(psutil.Process(pid).children(recursive=True))
    except psutil.NoSuchProcess:
        return ()


def _merge_processes(*groups: Sequence[psutil.Process]) -> tuple[psutil.Process, ...]:
    merged: dict[tuple[int, float], psutil.Process] = {}
    for group in groups:
        for process in group:
            try:
                key = (process.pid, process.create_time())
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                key = (process.pid, 0.0)
            merged[key] = process
    return tuple(merged.values())


def _signal_processes(processes: Sequence[psutil.Process], sig: signal.Signals) -> tuple[Exception, ...]:
    errors = []
    for process in processes:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            continue
        except Exception as exc:
            errors.append(exc)
    return tuple(errors)


def _wait_processes(processes: Sequence[psutil.Process], timeout: float) -> tuple[psutil.Process, ...]:
    if not processes:
        return ()
    try:
        _, alive = psutil.wait_procs(tuple(processes), timeout=timeout)
    except (psutil.Error, OSError):
        return tuple(processes)
    remaining = []
    for process in alive:
        try:
            if process.status() != psutil.STATUS_ZOMBIE:
                remaining.append(process)
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, OSError):
            remaining.append(process)
    return tuple(remaining)


def _signal_process_group(process_group: int, sig: signal.Signals) -> Exception | None:
    try:
        os.killpg(process_group, sig)
    except ProcessLookupError:
        return None
    except Exception as exc:
        return exc
    return None


def _termination_error_detail(errors: Sequence[Exception]) -> str:
    return 'process inspection failed: ' + '; '.join(
        str(error) or type(error).__name__
        for error in errors
    )


def _cleanup_worker_descendants(pid: int, timeout: float) -> None:
    descendants = _descendants(pid)
    errors = list(_signal_processes(descendants, signal.SIGTERM))
    alive = _wait_processes(descendants, timeout)
    errors.extend(_signal_processes(alive, signal.SIGKILL))
    alive = _wait_processes(alive, timeout)
    if alive or errors:
        detail = []
        if alive:
            detail.append('descendants ' + ','.join(str(item.pid) for item in alive))
        if errors:
            detail.append(_termination_error_detail(errors))
        raise ExecutionCleanupError(
            'operation left descendants that could not be cleaned: ' + '; '.join(detail),
            alive,
            unverified=bool(errors),
        )


async def _drain_output(stream: asyncio.StreamReader, *, limit: int = 64 * 1024) -> bytes:
    retained = bytearray()
    while chunk := await stream.read(8192):
        retained.extend(chunk)
        if len(retained) > limit:
            del retained[:-limit]
    return bytes(retained)


async def _forward_operation_events(sock: socket.socket, ctx: OperationContext) -> None:
    reader, writer = await asyncio.open_connection(
        sock=sock,
        limit=_EVENT_LIMIT,
    )
    try:
        while line := await reader.readline():
            if not line.endswith(b'\n'):
                _LOG.warning('isolated worker emitted an incomplete operation event')
                break
            try:
                data = json.loads(line)
                await ctx.record(
                    str(data['event_type']),
                    str(data.get('message') or ''),
                    level=data.get('level') or 'info',
                    status=data.get('status'),
                    data=data.get('data') or {},
                    current=data.get('current'),
                    total=data.get('total'),
                )
            except Exception as exc:
                _LOG.warning('isolated worker emitted an invalid operation event: %s', exc)
    finally:
        writer.close()
        await writer.wait_closed()


async def _consume_completion(completion: asyncio.Task[OperationResult | None]) -> None:
    try:
        await asyncio.shield(completion)
    except asyncio.CancelledError:
        if not completion.cancelled():
            raise
    except OperationExecutionError:
        return


def _validated_result(operation: Operation, result: object) -> OperationResult:
    if not isinstance(result, OperationResult):
        raise OperationExecutionError(f'{operation.spec.op_id} must return OperationResult')
    try:
        return result.validate_for(operation.spec)
    except (DefinitionError, TypeError) as exc:
        raise OperationExecutionError(
            f'{operation.spec.op_id} returned an invalid result: {exc}'
        ) from exc


class _EventWriter:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer

    @classmethod
    async def open(cls, event_fd: int) -> Self:
        sock = socket.socket(fileno=event_fd)
        sock.setblocking(False)
        sock.set_inheritable(False)
        try:
            _, writer = await asyncio.open_connection(sock=sock)
        except BaseException:
            sock.close()
            raise
        return cls(writer)

    async def __call__(self, event: OperationEvent) -> None:
        payload = json.dumps(
            {
                'event_type': event.event_type,
                'level': event.level,
                'status': event.status,
                'message': event.message,
                'data': dict(event.data),
                'current': event.current,
                'total': event.total,
            },
            ensure_ascii=False,
            separators=(',', ':'),
        ).encode() + b'\n'
        self._writer.write(payload)
        await self._writer.drain()

    async def close(self) -> None:
        self._writer.close()
        await self._writer.wait_closed()


async def _worker(request_path: Path, result_path: Path, event_fd: int) -> None:
    request = pickle.loads(request_path.read_bytes())
    if not isinstance(request, _IsolatedRequest):
        raise TypeError('isolated operation request has an invalid type')
    target: object = importlib.import_module(request.module)
    for part in request.qualname.split('.'):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f'{request.module}.{request.qualname} is not callable')
    operation: Operation = target  # type: ignore[assignment]
    reporter = await _EventWriter.open(event_fd)
    os.register_at_fork(after_in_child=lambda: _close_inherited_fd(event_fd))
    context = OperationContext(
        request.run_id,
        request.invocation_id,
        request.partition_key,
        reporter,
    )
    try:
        try:
            result = _validated_result(
                operation,
                await operation(context, **dict(request.inputs)),
            )
        finally:
            await asyncio.to_thread(
                _cleanup_worker_descendants,
                os.getpid(),
                request.cleanup_timeout,
            )
        temporary = result_path.with_suffix('.tmp')
        temporary.write_bytes(pickle.dumps(dict(result.values), protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(temporary, result_path)
    finally:
        await reporter.close()


def _close_inherited_fd(file_descriptor: int) -> None:
    try:
        os.close(file_descriptor)
    except OSError:
        pass


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    parser.add_argument('result', type=Path)
    parser.add_argument('event_fd', type=int)
    args = parser.parse_args()
    asyncio.run(_worker(args.request, args.result, args.event_fd))


__all__ = ['ExecutionCleanupError', 'ExecutionHandle', 'start_execution']
