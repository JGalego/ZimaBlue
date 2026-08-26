# Reconstruct a pool from phone views

A single photograph can hide a bad edge behind glare, a person or perspective.
Several independently rectified photographs expose those disagreements.
`pool_from_phones()` traces each view and keeps geometry supported by a quorum
of the views.

Place four visible markers around the pool as a rectangle and measure its width
and height. Photograph the whole pool from at least three positions. In every
image, list the same four marker corners in shared survey order—not merely
clockwise order in that image.

```python
import zimablue as zb

views = [
    zb.PhoneView(
        "north.jpg",
        corners=((122, 91), (1102, 180), (984, 744), (205, 698)),
        rectangle=(12.0, 7.0),
        sample=(540, 410),
    ),
    zb.PhoneView(
        "south.jpg",
        corners=((1088, 702), (173, 665), (250, 132), (1017, 96)),
        rectangle=(12.0, 7.0),
        sample=(590, 390),
    ),
]

pool = zb.pool_from_phones(
    views,
    depth=1.6,
    overlay_directory="survey-checks",
)
```

Inspect every saved overlay. Perspective calibration gives all traces a common
metric frame; majority fusion does not repair a wrongly labelled rectangle or
a segmentation that selected the sky.

## Measured depth

Surface photographs cannot reveal floor depth. Record depth at known survey
coordinates with a pole or sonar. When retaining the diagnostic reconstruction,
fit a plane explicitly:

```python
traces = [zb.trace_pool(view.image, sample=view.sample,
                        corners=(view.corners, view.rectangle))
          for view in views]
reconstruction = zb.fuse_phone_traces(traces)
depth = zb.fit_phone_depth(
    reconstruction.boundary,
    [
        zb.DepthObservation(1.0, 2.0, 1.1),
        zb.DepthObservation(6.0, 2.0, 1.7),
        zb.DepthObservation(11.0, 2.0, 2.2),
    ],
)
pool = reconstruction.pool(depth)
```

Three or more non-collinear locations identify a sloped plane. Equal readings
produce a constant-depth model. The fitter rejects non-positive or non-finite
depths instead of manufacturing a plausible floor from invalid measurements.

Use `fuse_phone_traces()` directly when an application already owns the
individual `PoolTrace` objects. Its result reports per-view intersection over
union, area variation, quorum, and warnings. A low agreement score is a reason
to repeat the survey, not a confidence value to hide.