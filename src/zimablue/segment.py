"""Segment Anything as a water mask, over onnxruntime.

Why this exists
---------------

The colour rules in :mod:`zimablue.imaging` know what water looks like and do
not know where it stops. They match a hue, and then have to be argued outward
pixel by pixel to reach the coping, because the last stretch of a pool is very
shallow water that grades continuously into the water beside it. SAM has the
opposite shape: prompted with one point inside a thing, it is very good at the
edge of that thing and has no idea which thing you meant. Prompted at the
middle of a pool it will happily return the whole sunlit terrace.

So they are used together. SAM proposes, the colour rule disposes: every
candidate mask SAM returns is scored against the pixels the colour rule thinks
are water, and the best agreement wins. Neither part is doing the other's job.

Where it pays
-------------

A photograph the colour rules struggle with -- a black-bottomed pool, a green
one, half of it in shade -- is exactly where a learned mask helps, and the
tracer's own hue rule is then a poor judge of the candidates. In that case the
ranking falls back to SAM's predicted IoU among the candidates containing the
seed point, and :attr:`SamSegmenter.candidates` says which rule was used.

Weights
-------

None are bundled and nothing is downloaded at import. MobileSAM is about 45 MB
across the two files and runs in a second or two on a CPU::

    huggingface-cli download Acly/MobileSAM \\
        mobile_sam_image_encoder.onnx sam_mask_decoder_multi.onnx --local-dir models/

    seg = SamSegmenter.load(
        "models/mobile_sam_image_encoder.onnx",
        "models/sam_mask_decoder_multi.onnx",
    )
    traced = zb.trace_pool("pool.jpg", sample=(700, 800), width=25.0, segmenter=seg)

Any SAM export following the reference ONNX scripts works, MobileSAM and
SAM-ViT alike. The decoder must be the *multi*-mask variant to have candidates
worth choosing between; a single-mask decoder is accepted and simply gives the
chooser one option.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zimablue.imaging import _water_by_sample, require_pillow

__all__ = ["ONNX_HINT", "Candidate", "SamSegmenter"]

ONNX_HINT = "onnxruntime is needed to run a SAM model. Install it with:  pip install 'zimablue[ml]'"

SIDE = 1024
"""The side SAM's image encoder works at. Not a tunable: it is baked into the
positional embeddings of every published checkpoint."""

PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

RECALL_WEIGHT = 2.0
"""How much more recall counts than precision when scoring candidates.

See :meth:`SamSegmenter._choose`. At 1.0 -- plain F1 -- the chooser clips the
steps off a pool, because the judge it is scoring against cannot see them
either.
"""

DECODER_INPUTS = (
    "image_embeddings",
    "point_coords",
    "point_labels",
    "mask_input",
    "has_mask_input",
    "orig_im_size",
)


@dataclass(frozen=True)
class Candidate:
    """One mask SAM offered, and how it scored."""

    index: int
    fraction: float
    """How much of the frame it covers."""

    predicted_iou: float
    """SAM's own confidence. Worth little for picking *which thing* you meant."""

    agreement: float
    """F2 against the colour rule's water, or ``nan`` when there was none."""

    contains_seed: bool

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Candidate({self.index}: {self.fraction:.1%} of frame, "
            f"iou={self.predicted_iou:.2f}, agreement={self.agreement:.2f})"
        )


class SamSegmenter:
    """A :class:`~zimablue.imaging.Segmenter` backed by a SAM ONNX export.

    Construct it with :meth:`load` (paths) or :meth:`from_env`, or hand it two
    session objects directly -- anything with ``get_inputs()`` and ``run()``,
    which is what makes this testable without 45 MB of weights.
    """

    def __init__(self, encoder: Any, decoder: Any, *, colour_judge: bool = True) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.colour_judge = colour_judge
        self.candidates: list[Candidate] = []
        """Scored candidates from the last call. Worth printing when a trace
        comes out wrong: it says whether SAM offered a good mask and the
        chooser missed it, or never offered one."""

        self.chosen: int = -1
        self.ranked_by: str = "unused"
        """``"agreement"`` or ``"predicted_iou"`` -- which rule picked the mask."""

        names = {i.name for i in decoder.get_inputs()}
        missing = [n for n in DECODER_INPUTS if n not in names]
        if missing:
            raise ValueError(
                f"this decoder is missing {', '.join(missing)}. zimablue expects the "
                "reference SAM ONNX export (sam_mask_decoder_multi.onnx and friends); "
                f"this one takes {', '.join(sorted(names))}"
            )

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        encoder: str | os.PathLike[str],
        decoder: str | os.PathLike[str],
        *,
        providers: list[str] | None = None,
        colour_judge: bool = True,
    ) -> SamSegmenter:
        """Load two ``.onnx`` files. CPU by default, which is fast enough."""
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the env
            raise ModuleNotFoundError(ONNX_HINT) from exc

        providers = providers or ["CPUExecutionProvider"]
        return cls(
            ort.InferenceSession(str(encoder), providers=providers),
            ort.InferenceSession(str(decoder), providers=providers),
            colour_judge=colour_judge,
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> SamSegmenter:
        """Load the paths in ``ZIMABLUE_SAM_ENCODER`` and ``ZIMABLUE_SAM_DECODER``.

        For the tests and examples that should run when the weights happen to
        be on the machine and skip when they are not.
        """
        encoder = os.environ.get("ZIMABLUE_SAM_ENCODER")
        decoder = os.environ.get("ZIMABLUE_SAM_DECODER")
        if not encoder or not decoder:
            raise RuntimeError(
                "set ZIMABLUE_SAM_ENCODER and ZIMABLUE_SAM_DECODER to the two .onnx "
                "files, or call SamSegmenter.load(encoder, decoder) with the paths"
            )
        return cls.load(encoder, decoder, **kwargs)

    # ------------------------------------------------------------------
    def __call__(self, rgb: NDArray[np.uint8], sample: tuple[int, int] | None) -> NDArray[np.bool_]:
        if sample is None:
            raise ValueError(
                "SAM needs a point to be prompted with. Pass sample=(x, y) with a "
                "pixel inside the pool"
            )
        rows, cols = rgb.shape[:2]
        if not (0 <= sample[0] < cols and 0 <= sample[1] < rows):
            raise ValueError(f"sample {sample} is outside the {cols}x{rows} image")

        embeddings = self._encode(rgb)
        masks, ious = self._decode(embeddings, sample, (rows, cols))
        return self._choose(masks, ious, rgb, sample)

    # ------------------------------------------------------------------
    def _encode(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Resize to SAM's 1024 px frame and run the image encoder.

        Two export conventions are in the wild and they are not distinguishable
        from the file name. One takes the image as ``HxWx3`` and normalises and
        pads inside the graph; the other takes a finished ``1x3x1024x1024``
        tensor. Neither *resizes*, which is a good way to lose an afternoon:
        the padding is a plain pad, so handing a 2000 px photo to the first
        kind silently crops it to its top-left corner and segments that.
        """
        spec = self.encoder.get_inputs()[0]
        resized = _resize_longest(rgb, SIDE)

        if len(spec.shape) == 3:
            feed = resized.astype(np.float32)
        else:
            normalised = (resized.astype(np.float32) - PIXEL_MEAN) / PIXEL_STD
            padded = np.zeros((SIDE, SIDE, 3), dtype=np.float32)
            padded[: resized.shape[0], : resized.shape[1]] = normalised
            feed = padded.transpose(2, 0, 1)[None]

        result = self.encoder.run(None, {spec.name: feed})[0]
        return np.asarray(result, dtype=np.float32)

    def _decode(
        self,
        embeddings: NDArray[np.float32],
        sample: tuple[int, int],
        size: tuple[int, int],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        rows, cols = size
        scale = SIDE / max(rows, cols)
        outputs = self.decoder.run(
            None,
            {
                "image_embeddings": embeddings,
                # The prompt lives in the resized frame, not the original one.
                "point_coords": np.array(
                    [[[sample[0] * scale, sample[1] * scale]]], dtype=np.float32
                ),
                "point_labels": np.array([[1.0]], dtype=np.float32),
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros(1, dtype=np.float32),
                # Asking for the mask back at the working resolution means the
                # rest of the tracer never learns that a neural net was here.
                "orig_im_size": np.array([rows, cols], dtype=np.float32),
            },
        )
        masks = np.asarray(outputs[0], dtype=np.float32)
        ious = np.asarray(outputs[1], dtype=np.float32).ravel()
        return masks[0], ious

    def _choose(
        self,
        masks: NDArray[np.float32],
        ious: NDArray[np.float32],
        rgb: NDArray[np.uint8],
        sample: tuple[int, int],
    ) -> NDArray[np.bool_]:
        """Pick the candidate that is the pool.

        SAM's own confidence cannot do this. It scores how cleanly each mask
        cuts out *a* thing, and the terrace is every bit as clean a thing as
        the water in it -- on the photo this was written against it rated the
        whole sunlit deck at 0.997.

        The colour rule can, but it is a biased judge and the bias has a
        direction: it under-detects shallow water, so it misses the steps and
        the last stretch before the coping. Scoring candidates by F1 therefore
        rewards the mask that stops short, and SAM's candidates are nested, so
        the mistake is always the same one -- it clips the shallow end off the
        pool. Weighting recall higher (F2) says what is actually meant: a
        candidate that misses water the colour rule is sure about is worse than
        one that includes water the colour rule was never going to find.
        """
        water = _water_by_sample(rgb, sample, 16.0, 0.22, 0.07) if self.colour_judge else None
        enough = water is not None and water.mean() > 0.005

        self.candidates = []
        for index in range(masks.shape[0]):
            mask = masks[index] > 0.0
            agreement = float("nan")
            if water is not None and mask.any() and water.any():
                precision = float(water[mask].mean())
                recall = float(mask[water].mean())
                total = RECALL_WEIGHT**2 * precision + recall
                agreement = (1 + RECALL_WEIGHT**2) * precision * recall / total if total else 0.0
            self.candidates.append(
                Candidate(
                    index=index,
                    fraction=float(mask.mean()),
                    predicted_iou=float(ious[index]) if index < len(ious) else float("nan"),
                    agreement=agreement,
                    contains_seed=bool(mask[sample[1], sample[0]]),
                )
            )

        # A candidate that does not contain the point it was prompted with is
        # SAM answering a question nobody asked.
        pool = [c for c in self.candidates if c.contains_seed] or self.candidates
        if enough:
            self.ranked_by = "agreement"
            best = max(pool, key=lambda c: (c.agreement, c.predicted_iou))
        else:
            # Nothing blue in the frame, which is the whole reason to be here.
            self.ranked_by = "predicted_iou"
            best = max(pool, key=lambda c: c.predicted_iou)

        self.chosen = best.index
        return np.asarray(masks[best.index] > 0.0, dtype=bool)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SamSegmenter(chosen={self.chosen}, ranked_by={self.ranked_by!r})"


def _resize_longest(rgb: NDArray[np.uint8], side: int) -> NDArray[np.uint8]:
    """Scale so the longest side is ``side``, keeping the aspect ratio."""
    require_pillow()
    from PIL import Image

    rows, cols = rgb.shape[:2]
    scale = side / max(rows, cols)
    if scale == 1.0:
        return rgb
    target = (max(round(cols * scale), 1), max(round(rows * scale), 1))
    return np.asarray(Image.fromarray(rgb).resize(target, Image.Resampling.BILINEAR))
