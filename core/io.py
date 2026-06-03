from __future__ import annotations

import re


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class StripAnsiWriter:
    def __init__(self, f):
        self._f = f

    def write(self, s):
        return self._f.write(_ANSI_RE.sub("", s))

    def flush(self):
        self._f.flush()

    def __getattr__(self, name):
        return getattr(self._f, name)
