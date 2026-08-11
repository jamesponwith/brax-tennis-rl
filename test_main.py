"""Parametrized tests are this repo's table-driven shape — see CLAUDE.md."""

import pytest

from main import greet


@pytest.mark.parametrize(
    ("name", "want"),
    [
        ("flywheel", "Hello, flywheel!"),
        ("", "Hello, !"),
    ],
)
def test_greet(name: str, want: str) -> None:
    assert greet(name) == want
