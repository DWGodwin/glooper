from fastapi import APIRouter

from server.config import get_enabled_plugins, get_plugin_config

router = APIRouter(prefix="/api")


@router.get("/config/plugins")
def list_plugins():
    return get_enabled_plugins()


@router.get("/config/plugins/{name}")
def plugin_config(name: str):
    return get_plugin_config(name)
