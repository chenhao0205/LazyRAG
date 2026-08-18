from __future__ import annotations

import os
import resource
import runpy
import sys
from collections.abc import Sequence


BLOCKED_EVENTS = {
    'os.exec', 'os.fork', 'os.forkpty', 'os.posix_spawn', 'os.spawn', 'os.system',
    'subprocess.Popen',
    'os.chdir', 'os.chmod', 'os.chown', 'os.link', 'os.mkdir', 'os.remove',
    'os.rename', 'os.rmdir', 'os.symlink', 'os.truncate', 'os.utime',
}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def guard(event: str, arguments: tuple[object, ...]) -> None:
    if event.startswith('socket.') or event in BLOCKED_EVENTS:
        raise RuntimeError(f'Demo operation blocked: {event}')
    if event != 'open':
        return
    mode = arguments[1] if len(arguments) > 1 else ''
    flags = arguments[2] if len(arguments) > 2 else 0
    if (isinstance(mode, str) and any(marker in mode for marker in 'wax+')) or (
        isinstance(flags, int) and flags & WRITE_FLAGS
    ):
        raise RuntimeError('Demo file writes are blocked')


def main(arguments: Sequence[str] | None = None) -> int:
    values = list(arguments if arguments is not None else sys.argv[1:])
    if len(values) != 4:
        raise RuntimeError('Demo runtime expects script, input, source and output limit')
    script, input_path, source_path, raw_limit = values
    limit = int(raw_limit)
    sys.dont_write_bytecode = True
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    except (OSError, ValueError):
        pass
    sys.addaudithook(guard)
    sys.path.insert(0, source_path)
    sys.argv = [script, '--input', input_path]
    runpy.run_path(script, run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = ['guard', 'main']
