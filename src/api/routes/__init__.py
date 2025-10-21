"""
API routes package.
"""

from .translation import router as translation_router
from .system import router as system_router

__all__ = [
    "translation_router",
    "system_router"
]