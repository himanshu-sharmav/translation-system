"""
API package for the machine translation backend system.
"""

from .app import create_app
from .models import *
from .routes import *

__all__ = [
    "create_app"
]