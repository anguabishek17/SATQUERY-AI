"""
Checks image count, modality, format, and pairing compatibility before any
model is invoked. Cheap and deterministic — fail fast with a clear message
rather than letting a specialist tool crash on bad input.
"""
from app.schemas import ImageRef, InputConfig
from app.config import ALLOWED_EXTENSIONS


class ValidationError(Exception):
    pass


def infer_input_config(images: list[ImageRef]) -> InputConfig:
    if len(images) == 1:
        return InputConfig.single

    if len(images) == 2:
        modalities = {img.modality for img in images}
        if modalities == {"optical", "sar"}:
            return InputConfig.cross_modal
        if modalities == {"optical"} or modalities == {"sar"}:
            return InputConfig.bi_temporal

    raise ValidationError(
        f"Unsupported input configuration: {len(images)} image(s) with "
        f"modalities {[img.modality for img in images]}. Expected 1 image, "
        f"or a co-registered optical+SAR pair, or a same-modality bi-temporal pair."
    )


def validate_format(file_path: str) -> None:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(f"Unsupported file format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")
