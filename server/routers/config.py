from fastapi import APIRouter

from server.config import get_enabled_plugins, get_plugin_config, get_vectorization_config

router = APIRouter(prefix="/api")


@router.get("/config/plugins")
def list_plugins():
    return get_enabled_plugins()


@router.get("/config/plugins/{name}")
def plugin_config(name: str):
    return get_plugin_config(name)


@router.get("/config/vectorization")
def vectorization_config():
    return get_vectorization_config()


@router.get("/config/vectorization/{label_class}")
def vectorization_config_for_class(label_class: str):
    return get_vectorization_config(label_class)
