"""Build hook: make the README work on PyPI as well as on GitHub.

GitHub renders the README *inside the repository*, so `docs/assets/replay.gif`
and `[MIT](LICENSE)` resolve against it. PyPI renders the same text standalone,
with no repository behind it, so every relative path is dead: images silently
fail to load and links 404.

Rather than keep two READMEs in sync, this rewrites the relative URLs to
absolute ones at build time. The file on disk stays repository-relative, which
is what GitHub wants; the copy in the package metadata is absolute, which is
what PyPI needs.

Two details worth knowing:

* **The animated logo is swapped for a still PNG.** PyPI's content security
  policy only allows images from its own camo proxy, and camo's handling of SVG
  is not something to rely on for the first thing a visitor sees.
* **The ref is configurable.** It defaults to ``main``, but the release
  workflow sets ``ZIMABLUE_DOCS_REF`` to the tag being published, so the page
  for version 1.2.3 shows the images as they were at v1.2.3 rather than
  whatever ``main`` drifts to later.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface

REPO = "JGalego/ZimaBlue"
DEFAULT_REF = "main"

# PyPI serves description images only through its camo proxy. GIF and PNG are
# safe; SVG is not worth the risk for the hero image.
STILL_IMAGES = {"docs/assets/logo-animated.svg": "docs/assets/logo.png"}

_IMG_SRC = re.compile(r'(<img\b[^>]*?\bsrc=")(?!https?://|data:)([^"]+)(")', re.IGNORECASE)
_MD_LINK = re.compile(r"\]\((?!https?://|mailto:|#)([^)]+)\)")


def absolutise(text: str, ref: str = DEFAULT_REF, repo: str = REPO) -> str:
    """Rewrite repository-relative URLs in Markdown to absolute ones."""
    raw = f"https://raw.githubusercontent.com/{repo}/{ref}/"
    blob = f"https://github.com/{repo}/blob/{ref}/"

    for animated, still in STILL_IMAGES.items():
        text = text.replace(animated, still)

    text = _IMG_SRC.sub(lambda m: f"{m.group(1)}{raw}{m.group(2)}{m.group(3)}", text)
    text = _MD_LINK.sub(lambda m: f"]({blob}{m.group(1)})", text)
    return text


class ReadmeHook(MetadataHookInterface):
    """Replaces the declared readme with a PyPI-safe rendering of it."""

    def update(self, metadata: dict[str, Any]) -> None:
        source = Path(self.root) / "README.md"
        ref = os.environ.get("ZIMABLUE_DOCS_REF", DEFAULT_REF).strip() or DEFAULT_REF
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": absolutise(source.read_text(encoding="utf-8"), ref=ref),
        }
