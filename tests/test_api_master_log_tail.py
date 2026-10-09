"""Regression tests for bounded task-log tail reads."""

import builtins
import collections
import os

import pytest

from ucagent.server.api_master import (
    _file_contains_any, _LOG_SCAN_CACHE, _tail_file, _tail_files, _task_stderr_tail,
)


@pytest.mark.parametrize("data", [
    b"", b"one", b"one\n", b"one\ntwo", b"one\ntwo\n",
    b"\n\n\n", b"one\r\ntwo\r\nthree", b"one\rtwo\rthree\r",
    b"one\vpart\ntwo\fpart\nthree\n", b"one\n\xff\ntwo\n",
    ("alpha\nb\u00e9ta\n\U0001f642\n" * 4000).encode("utf-8"),
    b"first\n" + b"x" * 20000 + b"\nlast",
    b"first\n" + b"x" * 20000 + b"\r\n" + b"z" * 8191,
], ids=[
    "empty", "single", "single-newline", "unterminated", "terminated",
    "blank-lines", "crlf", "cr", "embedded-controls", "invalid-utf8",
    "multibyte", "long-line", "block-boundary",
])
@pytest.mark.parametrize("limit", [0, 1, 2, 200])
def test_tail_matches_text_reader(tmp_path, data, limit):
    """Match the previous text-reader semantics across encodings and newlines."""
    path = tmp_path / "task.log"
    path.write_bytes(data)
    with path.open(encoding="utf-8", errors="replace") as stream:
        expected = "".join(collections.deque(stream, maxlen=limit))
    assert _tail_file(str(path), limit) == expected


def test_tail_files_preserves_order_and_unterminated_lines(tmp_path):
    """Count lines per file without inserting separators between files."""
    paths = [tmp_path / "first.log", tmp_path / "second.log"]
    paths[0].write_bytes(b"first\r\nunterminated")
    paths[1].write_bytes(b"second\nlast\n")
    for limit in (0, 1, 2, 3, 4, 200):
        expected = collections.deque(maxlen=limit)
        for path in paths:
            with path.open(encoding="utf-8", errors="replace") as stream:
                expected.extend(stream)
        assert _tail_files([str(path) for path in paths], limit) == "".join(expected)
    assert _tail_files([str(tmp_path / "missing.log"), ""], 200) == ""


def test_tail_reads_bounded_suffix_and_skips_earlier_files(tmp_path, monkeypatch):
    """A large log must not require a full scan or opening older filled-out logs."""
    older = tmp_path / "older.log"
    older.write_bytes(b"older\n")
    recent = tmp_path / "recent.log"
    recent.write_bytes(b"line\n" * 200000)
    real_open = builtins.open
    reads = []

    class TrackedReader:
        """Record actual bytes read while forwarding seek and context handling."""

        def __init__(self, stream):
            """Retain the real file stream."""
            self.stream = stream

        def __enter__(self):
            """Return the tracked stream."""
            return self

        def __exit__(self, *args):
            """Close the real stream."""
            return self.stream.__exit__(*args)

        def seek(self, *args):
            """Forward file positioning."""
            return self.stream.seek(*args)

        def read(self, size):
            """Record the number of bytes actually read."""
            data = self.stream.read(size)
            reads.append(len(data))
            return data

    def tracked_open(path, *args, **kwargs):
        """Reject unnecessary reads of the older file."""
        assert path == str(recent)
        return TrackedReader(real_open(path, *args, **kwargs))

    monkeypatch.setattr(builtins, "open", tracked_open)
    assert _tail_files([str(older), str(recent)]) == "line\n" * 200
    assert sum(reads) <= 8192


def test_error_scan_reads_only_appended_bytes(tmp_path, monkeypatch):
    """Unchanged logs read no bytes; appended marker fragments remain detectable."""
    path = tmp_path / "console.log"
    path.write_bytes(b"normal\n" * 200000)
    real_open = builtins.open
    reads = []

    class TrackedReader:
        """Forward file operations while tracking scan read sizes."""

        def __init__(self, stream):
            """Store the underlying file."""
            self.stream = stream

        def __enter__(self):
            """Expose the tracked reader."""
            return self

        def __exit__(self, *args):
            """Close the underlying file."""
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            """Forward non-read operations to the real file."""
            return getattr(self.stream, name)

        def read(self, size):
            """Record bytes consumed by the scan."""
            data = self.stream.read(size)
            reads.append(len(data))
            return data

    def tracked_open(*args, **kwargs):
        """Wrap scanner reads without changing writes."""
        return TrackedReader(real_open(*args, **kwargs))

    monkeypatch.setattr(builtins, "open", tracked_open)
    markers = ("AssertionError",)
    assert not _file_contains_any(str(path), markers)
    assert sum(reads) == path.stat().st_size
    reads.clear()
    assert not _file_contains_any(str(path), markers)
    assert reads == []
    for fragment, expected in [(b"Assertion", False), (b"Error\n", True)]:
        with path.open("ab") as stream:
            stream.write(fragment)
        reads.clear()
        assert _file_contains_any(str(path), markers) is expected
        assert sum(reads) <= len(fragment) + len(markers[0]) - 1
    with path.open("ab") as stream:
        stream.write(b"more output\n")
    reads.clear()
    assert _file_contains_any(str(path), markers)
    assert sum(reads) <= len(b"more output\n") + 2 * (len(markers[0]) - 1)


@pytest.mark.parametrize("change", ["truncate", "replace", "rewrite", "remove"])
def test_error_scan_invalidates_changed_logs(tmp_path, change):
    """Cached errors cannot survive observed file resets or missing files."""
    path = tmp_path / "console.log"
    path.write_bytes(b"AssertionError\n")
    markers = ("AssertionError",)
    assert _file_contains_any(str(path), markers)
    if change == "replace":
        replacement = tmp_path / "replacement.log"
        replacement.write_bytes(b"normal output\n" * 20)
        replacement.replace(path)
    elif change == "remove":
        path.unlink()
    else:
        old_mtime = path.stat().st_mtime_ns
        path.write_bytes(b"ok\n" if change == "truncate" else b"x" * 15)
        os.utime(path, ns=(old_mtime + 1000000000, old_mtime + 1000000000))
    assert not _file_contains_any(str(path), markers)
    path.write_bytes(b"AssertionError\n")
    assert _file_contains_any(str(path), markers)


def test_error_scan_boundaries_and_marker_sets(tmp_path):
    """Find split markers and keep different marker queries independent."""
    path = tmp_path / "console.log"
    path.write_bytes(b"x" * 8190 + b"AssertionError\n")
    assert _file_contains_any(str(path), ("AssertionError",))
    assert not _file_contains_any(str(path), ("Traceback",))
    assert not _file_contains_any(str(path), ())
    assert not _file_contains_any("", ("AssertionError",))
    for index in range(260):
        assert not _file_contains_any(str(path), (f"absent-{index}",))
    assert len(_LOG_SCAN_CACHE) <= 256


def test_stderr_display_retains_console_errors_after_append(tmp_path):
    """Display console errors after detection without exposing runtime scan state."""
    stderr = tmp_path / "stderr.log"
    console = tmp_path / "console.log"
    stderr.write_bytes(b"stderr\n")
    console.write_bytes(b"normal\n")
    task = {"stderr_log_path": str(stderr), "web_console_log_path": str(console)}
    assert _task_stderr_tail(task) == "stderr\n"
    with console.open("ab") as stream:
        stream.write(b"AssertionError\n")
    assert _task_stderr_tail(task) == "stderr\nnormal\nAssertionError\n"
    with console.open("ab") as stream:
        stream.write(b"finished\n")
    assert _task_stderr_tail(task, 1) == "finished\n"
    assert set(task) == {"stderr_log_path", "web_console_log_path"}
