"""The SAM segmenter, without 45 MB of weights.

Everything here runs against fake ONNX sessions. That is not a compromise:
the parts that actually broke while this was written were the resize, the
point transform and the choice between candidates -- all of them arithmetic
around the model rather than inside it, and all of them checkable exactly.
A test that downloaded a checkpoint would be slower and would check less.

One test does run the real thing, and skips unless ZIMABLUE_SAM_ENCODER and
ZIMABLUE_SAM_DECODER point at a pair of files.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zimablue.imaging import _water_by_sample, trace_pool
from zimablue.segment import DECODER_INPUTS, SIDE, SamSegmenter

pytest.importorskip("PIL")


class Spec:
    def __init__(self, name: str, shape: list) -> None:
        self.name = name
        self.shape = shape


class FakeEncoder:
    """Records what it was fed and returns an embedding of the right shape."""

    def __init__(self, rank: int = 3) -> None:
        self.shape = ["image_height", "image_width", 3] if rank == 3 else [1, 3, SIDE, SIDE]
        self.seen: np.ndarray | None = None

    def get_inputs(self):
        return [Spec("input_image", self.shape)]

    def run(self, _outputs, feed):
        self.seen = feed["input_image"]
        return [np.zeros((1, 256, 64, 64), dtype=np.float32)]


class FakeDecoder:
    """Returns fixed masks as logits, and records the prompt it was given."""

    def __init__(self, masks: np.ndarray, ious: list[float] | None = None) -> None:
        self.masks = np.asarray(masks, dtype=bool)
        self.ious = ious if ious is not None else [0.9] * len(self.masks)
        self.feed: dict | None = None

    def get_inputs(self):
        return [Spec(name, []) for name in DECODER_INPUTS]

    def run(self, _outputs, feed):
        self.feed = feed
        logits = np.where(self.masks, 5.0, -5.0).astype(np.float32)[None]
        return [logits, np.array([self.ious], dtype=np.float32), None]


def photo(size=(120, 80), water=None) -> np.ndarray:
    """Grey deck with a blue rectangle in it, at ``water`` or a default box."""
    cols, rows = size
    image = np.full((rows, cols, 3), 190, dtype=np.uint8)
    box = water if water is not None else (slice(20, 60), slice(30, 90))
    image[box] = (30, 120, 170)
    return image


# ----------------------------------------------------------------------
# Talking to the model
# ----------------------------------------------------------------------
def test_the_encoder_is_fed_an_image_scaled_to_sams_own_side():
    """The published encoders pad but never resize.

    Hand one a 2000 px photo and it crops to the top-left 1024 and segments
    that, confidently and wrongly. This is that afternoon, in a test.
    """
    rgb = photo((2000, 1500))
    encoder = FakeEncoder(rank=3)
    segmenter = SamSegmenter(encoder, FakeDecoder(np.ones((1, 1500, 2000), bool)))
    segmenter(rgb, (700, 800))

    assert encoder.seen is not None
    assert max(encoder.seen.shape[:2]) == SIDE
    assert encoder.seen.shape[:2] == (768, 1024), "aspect ratio should be kept"


def test_a_tensor_style_encoder_is_normalised_padded_and_transposed():
    rgb = photo((400, 200))
    encoder = FakeEncoder(rank=4)
    segmenter = SamSegmenter(encoder, FakeDecoder(np.ones((1, 200, 400), bool)))
    segmenter(rgb, (100, 100))

    fed = encoder.seen
    assert fed.shape == (1, 3, SIDE, SIDE)
    # 400x200 scales to 1024x512, so everything below row 512 is padding.
    assert np.all(fed[0, :, 512:, :] == 0.0)
    assert abs(float(fed[0, :, :512, :1024].mean())) < 2.0, "should be roughly zero-mean"


def test_the_prompt_is_transformed_into_the_resized_frame():
    rgb = photo((2000, 1000))
    decoder = FakeDecoder(np.ones((1, 1000, 2000), bool))
    SamSegmenter(FakeEncoder(), decoder)(rgb, (700, 800))

    scale = SIDE / 2000
    assert decoder.feed["point_coords"].ravel() == pytest.approx([700 * scale, 800 * scale])
    assert decoder.feed["point_labels"].ravel() == pytest.approx([1.0])
    # The mask comes back at the resolution the tracer is working at, so
    # nothing downstream has to know a model was involved.
    assert decoder.feed["orig_im_size"].ravel() == pytest.approx([1000, 2000])


def test_a_decoder_with_the_wrong_signature_says_so():
    class Wrong:
        def get_inputs(self):
            return [Spec("pixel_values", []), Spec("input_points", [])]

    with pytest.raises(ValueError, match="image_embeddings"):
        SamSegmenter(FakeEncoder(), Wrong())


def test_a_prompt_is_required():
    with pytest.raises(ValueError, match="sample"):
        SamSegmenter(FakeEncoder(), FakeDecoder(np.ones((1, 80, 120), bool)))(photo(), None)


def test_a_prompt_outside_the_image_says_so():
    segmenter = SamSegmenter(FakeEncoder(), FakeDecoder(np.ones((1, 80, 120), bool)))
    with pytest.raises(ValueError, match="outside"):
        segmenter(photo(), (500, 20))


# ----------------------------------------------------------------------
# Choosing between candidates
# ----------------------------------------------------------------------
def test_the_water_coloured_candidate_beats_the_bigger_one():
    """SAM rated the whole sunlit terrace at 0.997 on the photo this was
    written against. Its confidence says how cleanly a mask cuts out a thing,
    not whether that thing is the one you asked for."""
    rgb = photo()
    pool = np.zeros((80, 120), bool)
    pool[20:60, 30:90] = True
    everything = np.ones((80, 120), bool)

    segmenter = SamSegmenter(
        FakeEncoder(), FakeDecoder(np.stack([everything, pool]), ious=[0.99, 0.90])
    )
    mask = segmenter(rgb, (50, 40))

    assert segmenter.ranked_by == "agreement"
    assert segmenter.chosen == 1
    assert mask.sum() == pool.sum()


def test_the_more_inclusive_of_two_nested_pool_masks_wins():
    """A shallow end the colour rule can only half see.

    Columns 30-90 are deep water and obviously blue. 90-100 are the first
    step, still just blue enough to match. 100-140 are the rest of the steps,
    which the colour rule reads as coping and misses entirely. Scoring the
    candidates by F1 buys the precision of stopping at 90 and clips the whole
    shallow end off the pool; weighting recall keeps it.
    """
    rgb = photo(water=(slice(20, 60), slice(30, 100)))
    without = np.zeros((80, 120), bool)
    without[20:60, 30:90] = True
    with_steps = np.zeros((80, 120), bool)
    with_steps[20:60, 30:140] = True

    segmenter = SamSegmenter(
        FakeEncoder(), FakeDecoder(np.stack([without, with_steps]), ious=[0.99, 0.95])
    )
    segmenter(rgb, (50, 40))
    assert segmenter.chosen == 1, "should not clip the shallow end off the pool"

    water = _water_by_sample(rgb, (50, 40), 16.0, 0.22, 0.07)
    f1 = []
    for mask in (without, with_steps):
        precision, recall = float(water[mask].mean()), float(mask[water].mean())
        f1.append(2 * precision * recall / (precision + recall))
    assert f1[0] > f1[1], "F1 would have picked the clipped one -- that is the point"
    assert segmenter.candidates[1].agreement > segmenter.candidates[0].agreement


def test_a_candidate_that_misses_the_prompt_is_not_considered():
    rgb = photo()
    elsewhere = np.zeros((80, 120), bool)
    elsewhere[0:10, 0:10] = True
    pool = np.zeros((80, 120), bool)
    pool[20:60, 30:90] = True

    segmenter = SamSegmenter(
        FakeEncoder(), FakeDecoder(np.stack([elsewhere, pool]), ious=[1.0, 0.5])
    )
    segmenter(rgb, (50, 40))
    assert segmenter.chosen == 1
    assert [c.contains_seed for c in segmenter.candidates] == [False, True]


def test_without_anything_blue_it_falls_back_to_the_models_own_confidence():
    """A black-bottomed pool is the reason to reach for a learned mask, and
    also the reason the colour rule cannot referee it."""
    grey = np.full((80, 120, 3), 90, dtype=np.uint8)
    small = np.zeros((80, 120), bool)
    small[20:60, 30:90] = True
    big = np.ones((80, 120), bool)

    segmenter = SamSegmenter(FakeEncoder(), FakeDecoder(np.stack([big, small]), ious=[0.4, 0.95]))
    segmenter(grey, (50, 40))
    assert segmenter.ranked_by == "predicted_iou"
    assert segmenter.chosen == 1


def test_candidates_are_kept_for_inspection():
    rgb = photo()
    pool = np.zeros((80, 120), bool)
    pool[20:60, 30:90] = True
    segmenter = SamSegmenter(FakeEncoder(), FakeDecoder(np.stack([pool, np.ones((80, 120), bool)])))
    segmenter(rgb, (50, 40))

    assert len(segmenter.candidates) == 2
    assert segmenter.candidates[0].fraction == pytest.approx(pool.mean())
    assert "of frame" in repr(segmenter.candidates[0])


# ----------------------------------------------------------------------
# Through the tracer
# ----------------------------------------------------------------------
def test_a_segmenter_replaces_the_colour_rules_end_to_end():
    rgb = photo((200, 140))
    box = np.zeros((140, 200), bool)
    box[40:100, 50:150] = True  # 100 x 60 px

    traced = trace_pool(
        rgb,
        sample=(100, 70),
        width=10.0,  # so the 100 px width is 10 m
        segmenter=SamSegmenter(FakeEncoder(), FakeDecoder(box[None])),
        smooth_edges=0.0,
        close_gaps=0.0,
    )
    assert traced.area == pytest.approx(60.0, rel=0.05)


def test_growing_is_off_by_default_for_a_segmenter():
    """A mask that already reaches the coping does not want to be grown."""
    rgb = photo((200, 140))
    box = np.zeros((140, 200), bool)
    box[40:100, 50:150] = True

    def segmenter(_rgb, _sample):
        return box

    # Scale by metres_per_pixel, not width: pinning the width would hide the
    # growth by rescaling it away.
    grown = trace_pool(rgb, sample=(100, 70), metres_per_pixel=0.1, segmenter=segmenter, grow=6)
    plain = trace_pool(rgb, sample=(100, 70), metres_per_pixel=0.1, segmenter=segmenter)
    assert plain.area < grown.area


def test_a_segmenter_that_returns_the_wrong_shape_says_so():
    with pytest.raises(ValueError, match="mask for a"):
        trace_pool(
            photo(),
            sample=(50, 40),
            width=5.0,
            segmenter=lambda _rgb, _sample: np.ones((10, 10), bool),
        )


# ----------------------------------------------------------------------
@pytest.mark.skipif(
    not (os.environ.get("ZIMABLUE_SAM_ENCODER") and os.environ.get("ZIMABLUE_SAM_DECODER")),
    reason="set ZIMABLUE_SAM_ENCODER and ZIMABLUE_SAM_DECODER to run the real model",
)
def test_the_real_model_finds_a_synthetic_pool():
    pytest.importorskip("onnxruntime")
    rgb = photo((640, 480), water=(slice(120, 360), slice(160, 480)))
    traced = trace_pool(
        rgb, sample=(320, 240), width=16.0, segmenter=SamSegmenter.from_env(), close_gaps=0.0
    )
    assert traced.area == pytest.approx(12.0, rel=0.15)


# ----------------------------------------------------------------------
def test_from_env_says_which_variables_to_set(monkeypatch):
    monkeypatch.delenv("ZIMABLUE_SAM_ENCODER", raising=False)
    monkeypatch.delenv("ZIMABLUE_SAM_DECODER", raising=False)
    with pytest.raises(RuntimeError, match="ZIMABLUE_SAM_ENCODER"):
        SamSegmenter.from_env()


def test_an_image_already_at_sams_size_is_not_resampled():
    """Resizing 1024 to 1024 would be a lossy no-op."""
    from zimablue.segment import _resize_longest

    rgb = photo((SIDE, 400))
    assert _resize_longest(rgb, SIDE) is rgb
