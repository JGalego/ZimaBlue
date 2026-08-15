"""Playback: an interactive window, and headless exporters.

The interactive player is the fun one -- scrub, pause, change speed, watch the
little robot work.  The exporters exist because a lot of the time you are on a
remote box with no display and what you actually want is a GIF to paste into an
issue.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from zimablue.recording import Recording
from zimablue.replay.renderer import PALETTE, ReplayRenderer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = ["SPEEDS", "ReplayPlayer", "export_frames", "export_movie", "export_summary"]

SPEEDS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
"""Available playback rates. 25x is included because a 30-minute run at 10x is
still three minutes of watching, and sometimes you just want the shape of it."""

TARGET_FPS = 30.0


class ReplayPlayer:
    """An interactive matplotlib window with transport controls.

    Controls
    --------
    ``space``  pause / resume
    ``left`` / ``right``  step one second (hold shift for ten)
    ``up`` / ``down``  faster / slower
    ``r``  restart
    ``s``  save a PNG of the current frame
    ``q``  quit

    A slider along the bottom scrubs; dragging it pauses playback so the frame
    stays where you put it.
    """

    def __init__(
        self,
        recording: Recording,
        *,
        speed: float = 4.0,
        show_sensors: bool = True,
        start_paused: bool = False,
    ) -> None:
        self.recording = recording
        self.renderer = ReplayRenderer(recording, show_sensors=show_sensors)
        self.speed = float(speed)
        self.paused = start_paused
        self.index = 0
        self._scrubbing = False
        self._timer = None

        self.dt = float(recording.manifest.get("timestep", 0.02))
        self._build_controls()

    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        from matplotlib.widgets import Slider

        fig = self.renderer.fig
        fig.subplots_adjust(bottom=0.11)
        axis = fig.add_axes((0.06, 0.015, 0.72, 0.026), facecolor="#16222f")
        self.slider = Slider(
            axis,
            "",
            0,
            max(self.recording.n_frames - 1, 1),
            valinit=0,
            valstep=1,
            color=PALETTE["accent"],
            initcolor="none",
        )
        self.slider.valtext.set_visible(False)
        self.slider.on_changed(self._on_scrub)

        self._status = fig.text(
            0.80,
            0.019,
            "",
            color=PALETTE["ink"],
            fontsize=10,
            family="monospace",
        )
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh_status()

    # ------------------------------------------------------------------
    def _on_scrub(self, value: float) -> None:
        if self._scrubbing:
            return
        self.index = int(value)
        self.paused = True
        self.renderer.draw(self.index)
        self._refresh_status()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == " ":
            self.paused = not self.paused
        elif key in ("right", "shift+right"):
            self._seek(self.index + int((10 if "shift" in key else 1) / self.dt))
        elif key in ("left", "shift+left"):
            self._seek(self.index - int((10 if "shift" in key else 1) / self.dt))
        elif key == "up":
            self._change_speed(+1)
        elif key == "down":
            self._change_speed(-1)
        elif key == "r":
            self._seek(0)
            self.paused = False
        elif key == "s":
            path = Path(f"zimablue-frame-{self.index:06d}.png")
            self.renderer.fig.savefig(path, dpi=140, facecolor=PALETTE["panel"])
            print(f"saved {path}")
        elif key == "q":
            import matplotlib.pyplot as plt

            plt.close(self.renderer.fig)
            return
        self._refresh_status()

    def _change_speed(self, direction: int) -> None:
        current = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - self.speed))
        self.speed = SPEEDS[int(np.clip(current + direction, 0, len(SPEEDS) - 1))]

    def _seek(self, index: int) -> None:
        self.index = int(np.clip(index, 0, self.recording.n_frames - 1))
        self._scrubbing = True
        self.slider.set_val(self.index)
        self._scrubbing = False
        self.renderer.draw(self.index)

    def _refresh_status(self) -> None:
        state = "paused" if self.paused else "playing"
        self._status.set_text(f"{self.speed:g}x  {state}")

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        if not self.paused:
            # Advance by however many recorded frames fit in one wall-clock
            # frame at the current speed, so 10x really is ten times faster
            # rather than ten times choppier.
            step = max(1, round(self.speed / (TARGET_FPS * self.dt)))
            nxt = self.index + step
            if nxt >= self.recording.n_frames:
                nxt = 0
            self._seek(nxt)
            self._refresh_status()

    def show(self) -> None:
        """Open the window and block until it is closed."""
        import matplotlib.pyplot as plt

        self.renderer.draw(0)
        self._timer = self.renderer.fig.canvas.new_timer(interval=int(1000 / TARGET_FPS))
        self._timer.add_callback(self._tick)
        self._timer.start()
        plt.show()


# ----------------------------------------------------------------------
# Headless exporters
# ----------------------------------------------------------------------
def _frame_indices(recording: Recording, speed: float, fps: float) -> Sequence[int]:
    dt = float(recording.manifest.get("timestep", 0.02))
    step = max(1, round(speed / (fps * dt)))
    return range(0, recording.n_frames, step)


def export_movie(
    recording: Recording,
    path: str | Path,
    *,
    speed: float = 60.0,
    fps: float = 25.0,
    dpi: int = 90,
    show_sensors: bool = True,
) -> Path:
    """Render the run to an animated GIF or MP4.

    Chooses the writer from the file extension. MP4 needs ffmpeg; GIF works
    anywhere matplotlib does, which is why the CLI defaults to it.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.animation import FFMpegWriter, PillowWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = ReplayRenderer(recording, show_sensors=show_sensors, dpi=dpi)
    indices = list(_frame_indices(recording, speed, fps))

    writer = (
        FFMpegWriter(fps=int(fps), bitrate=2400)
        if path.suffix.lower() in (".mp4", ".mov", ".webm")
        else PillowWriter(fps=int(fps))
    )
    with writer.saving(renderer.fig, str(path), dpi):
        for index in indices:
            renderer.draw(index)
            writer.grab_frame(facecolor=PALETTE["panel"])
    import matplotlib.pyplot as plt

    plt.close(renderer.fig)
    return path


def export_frames(
    recording: Recording,
    directory: str | Path,
    *,
    count: int = 6,
    dpi: int = 110,
) -> list[Path]:
    """Write ``count`` evenly-spaced stills -- handy for a README or an issue."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    renderer = ReplayRenderer(recording, dpi=dpi)
    written: list[Path] = []
    for i, index in enumerate(np.linspace(0, recording.n_frames - 1, count).astype(int)):
        renderer.draw(int(index))
        out = directory / f"frame_{i:02d}.png"
        renderer.fig.savefig(out, dpi=dpi, facecolor=PALETTE["panel"])
        written.append(out)
    plt.close(renderer.fig)
    return written


def export_summary(recording: Recording, path: str | Path, *, dpi: int = 110) -> Path:
    """A four-panel post-run summary.

    The panels are chosen to make the coverage-versus-cleanliness distinction
    impossible to miss: the path the robot drove, the cells it visited, the
    dirt it started with, and the dirt it left.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from zimablue.replay.renderer import load_scene

    scene = load_scene(recording)
    spatial = recording.spatial
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi, facecolor=PALETTE["panel"])
    extent = scene.grid.extent
    navigable = scene.navigable

    def style(ax, title: str) -> None:
        ax.set_title(title, color=PALETTE["ink"], fontsize=11, family="monospace")
        ax.set_facecolor(PALETTE["panel"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        outline = np.asarray(scene.pool.boundary.exterior.coords)
        ax.plot(outline[:, 0], outline[:, 1], color=PALETTE["coping"], linewidth=2)

    # Path
    ax = axes[0][0]
    style(ax, "path driven")
    ax.plot(
        recording.frames["x"],
        recording.frames["y"],
        color=PALETTE["trail"],
        linewidth=0.7,
        alpha=0.85,
    )

    # Visits
    ax = axes[0][1]
    style(ax, "coverage (visit count)")
    visits = spatial.get("visits")
    if visits is not None:
        ax.imshow(
            np.where(navigable, visits, np.nan),
            extent=extent,
            origin="lower",
            cmap="cividis",
            interpolation="nearest",
        )

    # Dirt before / after, on a shared scale so the panels are comparable.
    initial = spatial.get("initial_dirt")
    remaining = spatial.get("remaining_dirt")
    vmax = float(np.nanpercentile(initial[navigable], 98)) if initial is not None else 1.0
    for ax, data, title in (
        (axes[1][0], initial, "dirt at start"),
        (axes[1][1], remaining, "dirt at end"),
    ):
        style(ax, title)
        if data is not None:
            ax.imshow(
                np.where(navigable, data, np.nan),
                extent=extent,
                origin="lower",
                cmap="copper_r",
                vmin=0,
                vmax=vmax,
                interpolation="bilinear",
            )

    metrics = recording.metrics
    fig.suptitle(
        f"coverage {metrics.get('coverage', 0) * 100:.0f}%    "
        f"dirt removed {metrics.get('dirt_removed_fraction', 0) * 100:.0f}%    "
        f"{metrics.get('runtime', 0) / 60:.0f} min",
        color=PALETTE["ink"],
        family="monospace",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=PALETTE["panel"])
    plt.close(fig)
    return path
