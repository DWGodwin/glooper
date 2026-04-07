from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from server.config import get_config

router = APIRouter(prefix="/api/models")


@router.get("/sam-decoder")
def sam_decoder():
    cfg = get_config()
    path = Path(cfg["data_dir"]) / "sam_decoder.onnx"
    if not path.exists():
        raise HTTPException(status_code=404, detail="sam_decoder.onnx not found")
    return FileResponse(path, media_type="application/octet-stream")
