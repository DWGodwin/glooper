import importlib
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import get_enabled_plugins, validate_plugin_deps
from server.db import init_db
from server.providers import init_provider
from server.routers import chips, config, labels, models, study_areas

logger = logging.getLogger(__name__)

app = FastAPI(title="Glooper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(config.router)
app.include_router(study_areas.router)
app.include_router(chips.router)
app.include_router(labels.router)
app.include_router(models.router)


for plugin_name in get_enabled_plugins():
    try:
        mod = importlib.import_module(f"plugins.{plugin_name}.server.router")
        app.include_router(mod.router)
        logger.info("Loaded plugin router: %s", plugin_name)
    except Exception:
        logger.exception("Failed to load plugin '%s'", plugin_name)


@app.on_event("startup")
def startup():
    init_db()
    init_provider()
    validate_plugin_deps()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
