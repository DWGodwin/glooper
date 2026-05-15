import threading

import pytest
import yaml


def _reset_singletons():
    """Clear cached config and DB state so each test starts clean."""
    import server.config
    import server.db
    import server.providers

    server.config._config = None
    server.db._db_path = None
    server.db._local = threading.local()
    server.providers._png_dir = None


@pytest.fixture
def test_config(tmp_path, monkeypatch):
    """Write a minimal test config and point GLOOPER_CONFIG at it.

    Resets cached singletons in server.config and server.db so the test config
    is the one that actually takes effect.
    """
    cfg = {
        "chip_size_m": 67.2,
        "chip_size_px": 448,
        "crs": "EPSG:32619",
        "max_chips_per_request": 100,
        "db_path": str(tmp_path / "test.duckdb"),
        "data_dir": str(tmp_path / "data"),
        "worker_host": "nonexistent.invalid",
        "worker_port": 9100,
        "plugins": [],
        "vectorization": {
            "strategy": "simplify",
            "tolerance_px": 0.5,
            "min_area_px": 4,
            "max_vertices": None,
            "class_overrides": {},
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("GLOOPER_CONFIG", str(cfg_path))

    _reset_singletons()
    yield cfg
    _reset_singletons()


@pytest.fixture
def db(test_config):
    """Initialize an empty DuckDB at the test config's db_path."""
    from server.db import init_db

    init_db()
    return test_config


@pytest.fixture
def client(test_config, monkeypatch):
    """FastAPI TestClient with worker calls stubbed out.

    Patches the worker_client symbols *as imported by each router module* so
    endpoints never actually open a TCP connection. The startup hook also
    needs check_worker patched on server.providers.
    """
    from fastapi.testclient import TestClient

    def _start_batch_stub(job_id, chips, crs):
        return job_id

    def _get_job_status_stub(job_id):
        return {
            "phase": "chips",
            "chips_total": 0,
            "chips_done": 0,
            "chips_failed": 0,
            "embed_total": 0,
            "embed_done": 0,
            "embed_failed": 0,
        }

    # Patch before importing app so startup hook is safe to run.
    import server.providers
    import server.routers.study_areas as study_areas_router

    monkeypatch.setattr(server.providers, "check_worker", lambda: True)
    monkeypatch.setattr(study_areas_router, "start_batch", _start_batch_stub)
    monkeypatch.setattr(study_areas_router, "get_job_status", _get_job_status_stub)

    from server.main import app

    with TestClient(app) as c:
        yield c
