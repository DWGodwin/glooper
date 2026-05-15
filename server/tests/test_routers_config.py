"""Endpoint tests for /api/config/*."""


def test_get_plugins_default_is_empty(client):
    resp = client.get("/api/config/plugins")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_vectorization_strips_class_overrides(client):
    resp = client.get("/api/config/vectorization")
    assert resp.status_code == 200
    body = resp.json()
    # Class overrides are internal merge machinery and shouldn't leak to the client
    assert "class_overrides" not in body
    # And the base config fields should be present
    assert body["strategy"] == "simplify"
    assert body["tolerance_px"] == 0.5
    assert body["min_area_px"] == 4


def test_get_vectorization_unknown_class_falls_back_to_base(client):
    resp = client.get("/api/config/vectorization/no_such_class")
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "simplify"
    assert body["tolerance_px"] == 0.5


def test_get_vectorization_applies_class_override(client):
    import server.config

    # Inject a class override into the live config and verify it's reflected.
    server.config._config["vectorization"]["class_overrides"] = {
        "solar_panel": {"strategy": "convex_hull", "tolerance_px": 2.0},
    }

    resp = client.get("/api/config/vectorization/solar_panel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "convex_hull"
    assert body["tolerance_px"] == 2.0
    # Non-overridden fields stay at base
    assert body["min_area_px"] == 4


def test_get_plugin_config_unknown_returns_empty(client):
    resp = client.get("/api/config/plugins/nonexistent_plugin")
    assert resp.status_code == 200
    assert resp.json() == {}
