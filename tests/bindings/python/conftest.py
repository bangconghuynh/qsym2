import pytest


@pytest.fixture(autouse=True)
def run_in_tmp_path(tmp_path, monkeypatch):
    """Runs every test inside its own temporary directory."""
    monkeypatch.chdir(tmp_path)
