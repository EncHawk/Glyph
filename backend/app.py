from pathlib import Path

import modal

ROOT_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libcairo2-dev",
        "libpango1.0-dev",
        "pkg-config",
        "poppler-utils",
        "tesseract-ocr",
        "shared-mime-info",
    )
    .pip_install_from_pyproject(str(ROOT_DIR / "pyproject.toml"))
    .env({"PYTHONPATH": str(ROOT_DIR / "src")})
)

volume = modal.Volume.from_name("glyph-data", create_if_missing=True)
VOLUME_PATH = "/glyph-data"

secrets = [
    modal.Secret.from_dotenv(str(ROOT_DIR / ".env")),
]

app = modal.App("glyph-backend", image=image, secrets=secrets)
