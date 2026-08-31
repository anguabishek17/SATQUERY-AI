import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import UPLOAD_DIR
from app.controller.validator import ValidationError, validate_format
from app.services.image_io import saved_path

router = APIRouter()

_MEDIA_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".tif": "image/tiff", ".tiff": "image/tiff",
}


@router.post("")
async def upload_image(file: UploadFile, modality: str, acquisition_date: str | None = None):
    """
    Stores an uploaded image and returns a file_id the frontend threads through
    to /api/query. modality must be "optical" or "sar".
    """
    if modality not in ("optical", "sar"):
        raise HTTPException(400, "modality must be 'optical' or 'sar'")

    try:
        validate_format(file.filename)
    except ValidationError as e:
        raise HTTPException(400, str(e))

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    dest = UPLOAD_DIR / f"{file_id}{ext}"
    dest.write_bytes(await file.read())

    return {
        "file_id": file_id,
        "modality": modality,
        "acquisition_date": acquisition_date,
        "filename": file.filename,
    }


@router.get("/{file_id}/preview")
def get_preview(file_id: str):
    """
    Serves the raw uploaded file so the frontend's map/image viewer can
    actually render it (previously the dashboard hardcoded previewUrl=null
    and the viewer stayed permanently blank — this is what fixes that).
    For plain JPG/PNG this is a direct passthrough; for TIFF the browser
    can't render it natively, so swap this for image_io.read_preview_png's
    georeferenced-PNG path once GeoTIFFs are wired in.
    """
    try:
        path = saved_path(file_id)
    except FileNotFoundError:
        raise HTTPException(404, "file not found")

    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type)
