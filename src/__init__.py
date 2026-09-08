"""Ashen Researcher.

Loads ``.env`` on package import so every entry point -- uvicorn, streamlit,
pytest, a bare ``python -c`` -- sees the same configuration without each one
having to remember to call ``load_dotenv()``.

This runs before any module-level ``os.getenv`` in the package, because those
modules are imported through this package. Real environment variables always
win: ``load_dotenv`` does not override what is already set, so
``$env:ASHEN_PIPELINE="stub"`` in a shell still beats the file.
"""

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # pragma: no cover - python-dotenv is optional at runtime
    # Without it, configuration still works through real environment variables.
    pass
