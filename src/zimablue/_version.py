"""Single source of truth for the ZimaBlue version.

Kept in its own module so that the recording layer can stamp runs with the
producing version without importing the whole package.
"""

from __future__ import annotations

__version__ = "0.3.0"
