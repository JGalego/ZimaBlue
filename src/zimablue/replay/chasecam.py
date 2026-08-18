"""Chase cam -- the view from just behind the cleaner.

The other two views each hide something.  From above you see the whole pool and
a robot the size of a postage stamp, so you read the *path* and lose the
machine.  From the bumper you see what the machine sees and never the machine
itself, so you read the *dirt* and lose all sense of what is doing the work.
The chase cam is the one that shows both: a metre back and half a metre up,
close enough that the brushes are visible and far enough that you can see the
clean stripe opening up behind.

It is the same floor renderer as
:mod:`~zimablue.replay.dirtcam` -- inverse perspective mapping, no 3D engine --
with the camera unbolted from the robot and floated behind it.  What that buys
is that the robot is now *in front of the camera*, so it has to be drawn, and
what gets drawn is the cleaner's own
:class:`~zimablue.robot.design.CleanerDesign`.  A quad-brush commercial machine
and a domed suction unit look like different machines from back here, because
they are.

The camera follows with lag.  Rigidly bolting it a metre behind makes a turn
look like the *pool* rotating, which is disorienting and wrong -- the robot is
what is turning. Letting the camera's heading chase the robot's means a turn
reads as the robot swinging out to one side of frame and the camera easing
after it, which is what a diver following it would see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from zimablue.replay._deps import require_matplotlib
from zimablue.replay.floorcam import FloorCamConfig, FloorCamera
from zimablue.replay.floorcam import rgb as _rgb
from zimablue.replay.renderer import PALETTE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zimablue.recording import Recording

__all__ = [
    "ChaseCam",
    "ChaseCamConfig",
    "export_chasecam",
    "export_chasecam_frames",
    "render_chasecam",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ChaseCamConfig(FloorCamConfig):
    """Where the following camera sits.

    The defaults put a 42 cm robot at roughly a third of the frame height, with
    a couple of metres of floor visible past it -- close enough to see the
    brushes turning, wide enough to see the swath it is leaving.
    """

    width: int = 384
    height: int = 216

    distance: float = 0.95
    """Metres behind the robot.

    Close enough that a 42 cm machine fills about a third of the frame width.
    Further back and it becomes a detail in a picture of a floor."""

    camera_height: float = 0.62
    """Metres above the floor. High enough to look over the machine at the
    floor beyond it, low enough that it is not just the top-down view again."""

    pitch: float = 0.34
    """Downward tilt in radians. Puts the horizon a third of the way down."""

    fov: float = 1.30
    """Horizontal field of view in radians (~75 degrees). Narrower than the
    bumper camera: a wide lens this far back makes the robot tiny and bends the
    pool walls."""

    far: float = 6.0
    """Further than the bumper camera can see, because the camera is higher and
    a higher camera gets more useful metres before the floor compresses."""

    lag: float = 0.22
    """Seconds of heading lag. 0 bolts the camera to the robot.

    This is the single setting that decides whether the view feels like a chase
    or like the pool spinning. A fifth of a second is enough for a turn to read
    as the robot swinging across frame, and short enough that the camera does
    not sail off on its own during a long arc.
    """

    shadow: float = 0.45
    """How dark the robot's contact shadow is, 0-1.

    Without one the machine looks pasted onto the floor rather than sitting on
    it -- the cheapest possible grounding cue, and the only lighting model
    here.
    """


class ChaseCam(FloorCamera):
    """The pool floor from behind the cleaner, with the cleaner in shot."""

    def __init__(self, recording: Recording, config: ChaseCamConfig | None = None) -> None:
        super().__init__(recording, config or ChaseCamConfig())
        self._smoothed: FloatArray | None = None

    # ------------------------------------------------------------------
    @property
    def cfg(self) -> ChaseCamConfig:
        assert isinstance(self.config, ChaseCamConfig)
        return self.config

    def _heading_track(self) -> FloatArray:
        """The camera's heading over the whole run, lagged behind the robot's.

        Computed once for the run rather than per frame, because a lag filter
        is a recurrence: evaluating it for frame 900 alone would need the 899
        before it anyway, and doing that per frame turns a scrub into an O(n^2)
        wait. Unwrapped before filtering, or every pass through +/-pi produces
        a camera that whips the long way round.
        """
        if self._smoothed is not None:
            return self._smoothed
        heading = np.unwrap(np.asarray(self.recording.frames["heading"], dtype=float))
        lag = self.cfg.lag
        if lag <= 0:
            self._smoothed = heading
            return heading

        dt = max(self.recording.frame_dt, 1e-6)
        alpha = float(np.clip(dt / (lag + dt), 1e-4, 1.0))
        smoothed = np.empty_like(heading)
        value = heading[0]
        for i, target in enumerate(heading):
            value += alpha * (target - value)
            smoothed[i] = value
        self._smoothed = smoothed
        return smoothed

    def camera_pose(self, index: int) -> tuple[float, float, float]:
        """Behind the robot, on the lagged heading."""
        index = int(np.clip(index, 0, self.recording.n_frames - 1))
        x, y, _ = self.robot_pose(index)
        yaw = float(self._heading_track()[index])
        return (x - np.cos(yaw) * self.cfg.distance, y - np.sin(yaw) * self.cfg.distance, yaw)

    # ------------------------------------------------------------------
    def draw_overlays(self, image: FloatArray, index: int) -> None:
        self._draw_robot(image, index)

    def _draw_robot(self, image: FloatArray, index: int) -> None:
        """The cleaner, from its design, projected into the frame.

        Each part is drawn twice: once flattened onto the floor as a shadow,
        once at its own height. That is not a lighting model and does not
        pretend to be -- it is two polygon fills, and it is the difference
        between a machine sitting on the floor and a sticker floating over it.
        """
        cfg = self.cfg
        scene = self.scene
        x, y, heading = self.robot_pose(index)
        cx, cy, yaw = self.camera_pose(index)
        design = scene.design
        length, width = scene.robot_length, scene.robot_width
        tall = scene.robot_height * design.dome

        def screen(outline: FloatArray, lift: float) -> tuple[FloatArray, FloatArray] | None:
            world = design.place(outline, length, width, x, y, heading)
            ahead, lateral = self.to_camera(world[:, 0], world[:, 1], cx, cy, yaw)
            # Anything at or behind the lens cannot be projected; with the
            # camera a metre back the robot never is, but a caller who sets
            # distance to zero should get nothing rather than a torn polygon.
            if np.any(ahead < 0.05):
                return None
            return self.project(ahead, lateral, height=lift)

        # Contact shadow: the hull, flat on the floor, slightly spread.
        shadow = screen(np.asarray(design.body) * 1.06, 0.0)
        if shadow is not None:
            self.fill_polygon(
                image, shadow[0], shadow[1], np.zeros(3), cfg.distance, alpha=cfg.shadow, fade=0.9
            )

        for part in design.drawable():
            lift = tall * (part.lift if part.name != "hull" else 0.0)
            # The hull is extruded: its floor outline and its top outline are
            # both drawn, plus the side wall between them, so the machine has
            # thickness instead of being a decal.
            if part.name == "hull":
                self._draw_extrusion(image, screen, part, tall * 0.55)
                continue
            projected = screen(np.asarray(part.outline), lift)
            if projected is None:
                continue
            self.fill_polygon(
                image,
                projected[0],
                projected[1],
                np.array(_rgb(part.colour)),
                cfg.distance,
                alpha=part.alpha,
                fade=self._lit(lift, tall),
            )

    def _draw_extrusion(self, image: FloatArray, screen: Any, part: Any, tall: float) -> None:
        """The hull as a solid: sides first, then the top face."""
        cfg = self.cfg
        outline = np.asarray(part.outline)
        base = screen(outline, 0.0)
        top = screen(outline, tall)
        if base is None or top is None:
            return

        # Side walls, one quad per edge, darker than the top so the silhouette
        # has an edge to it.
        side = np.array(_rgb(part.colour)) * 0.62
        n = len(outline)
        for i in range(n):
            j = (i + 1) % n
            cols = np.array([base[0][i], base[0][j], top[0][j], top[0][i]])
            rows = np.array([base[1][i], base[1][j], top[1][j], top[1][i]])
            self.fill_polygon(image, cols, rows, side, cfg.distance, fade=self._lit(0.0, 1.0))
        self.fill_polygon(
            image,
            top[0],
            top[1],
            np.array(_rgb(part.colour)),
            cfg.distance,
            fade=self._lit(tall, tall),
        )

    def _lit(self, lift: float, tall: float) -> float:
        """A flat blend factor, brighter for the parts nearer the surface.

        Underwater light comes from straight up, so a top face is lit and a
        side face is not. One multiply, no normals.
        """
        share = 0.0 if tall <= 0 else float(np.clip(lift / tall, 0.0, 1.0))
        return float(np.clip(0.80 + 0.20 * share, 0.0, 1.0))


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------
def render_chasecam(
    recording: Recording,
    index: int,
    *,
    ax: Any = None,
    camera: ChaseCam | None = None,
    config: ChaseCamConfig | None = None,
) -> Any:
    """Draw the chase view for one frame onto ``ax``."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    cam = camera or ChaseCam(recording, config)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=PALETTE["panel"])
    ax.clear()
    ax.imshow(cam.frame(index), interpolation="bilinear", aspect="auto")
    ax.set_axis_off()
    return ax


def export_chasecam_frames(
    recording: Recording,
    path: str | Path,
    *,
    count: int = 4,
    config: ChaseCamConfig | None = None,
    dpi: int = 130,
) -> Path:
    """A contact sheet of ``count`` chase views spread across the run."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cam = ChaseCam(recording, config)
    indices = np.linspace(0, recording.n_frames - 1, count).astype(int)
    fig, axes = plt.subplots(count, 1, figsize=(7.2, 4.05 * count), facecolor=PALETTE["panel"])
    axes = np.atleast_1d(axes)
    for ax, index in zip(axes, indices, strict=True):
        ax.imshow(cam.frame(int(index)), interpolation="bilinear", aspect="auto")
        ax.set_axis_off()
        ax.set_title(
            f"t = {recording.frames['time'][index]:.0f} s",
            color=PALETTE["ink"],
            fontsize=10,
            family="monospace",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=PALETTE["panel"])
    plt.close(fig)
    return path


def export_chasecam(
    recording: Recording,
    path: str | Path,
    *,
    speed: float = 60.0,
    fps: int = 20,
    dpi: int = 80,
    config: ChaseCamConfig | None = None,
) -> Path:
    """Render the run as a chase-cam animation.

    ``speed=1.0`` writes a file exactly as long as the run, as everywhere else.
    """
    require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib import animation

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cam = ChaseCam(recording, config)
    dt = max(recording.frame_dt, 1e-6)
    stride = max(speed / (fps * dt), 1.0)
    indices = [int(i * stride) for i in range(max(int(recording.n_frames / stride), 1))]

    figure, ax = plt.subplots(figsize=(7.2, 4.05), facecolor=PALETTE["panel"])
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    canvas = ax.imshow(cam.frame(0), interpolation="bilinear", aspect="auto")

    def update(index: int) -> tuple[Any, ...]:
        canvas.set_data(cam.frame(index))
        return (canvas,)

    anim = animation.FuncAnimation(figure, update, frames=indices, interval=1000 / fps, blit=True)
    writer = "ffmpeg" if str(path).endswith(".mp4") else "pillow"
    anim.save(str(path), writer=writer, dpi=dpi, savefig_kwargs={"facecolor": PALETTE["panel"]})
    plt.close(figure)
    return path
