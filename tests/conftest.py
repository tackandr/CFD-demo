"""Shared pytest configuration.

Collects short verification reports and prints them in the terminal summary,
so the ADR 0005 diagnostics are visible on a passing run.
"""

from __future__ import annotations

import pytest

_REPORTS = pytest.StashKey[list]()


@pytest.fixture(scope="session")
def report(pytestconfig):
    """Return a callable that records a line for the terminal summary."""
    lines = pytestconfig.stash.setdefault(_REPORTS, [])
    return lines.append


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    lines = config.stash.get(_REPORTS, [])
    if not lines:
        return
    terminalreporter.write_sep("=", "verification report")
    for line in lines:
        terminalreporter.write_line(line)
