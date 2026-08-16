"""The ``zimablue`` command line.

Six verbs: ``demo``, ``run``, ``replay``, ``batch``, ``inspect``, ``list``.

Errors are meant to be actionable -- an unknown preset lists the valid ones, a
missing recording says where to make one -- because the most common CLI failure
is a typo, and a stack trace is a poor answer to a typo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from zimablue._version import __version__

app = typer.Typer(
    name="zimablue",
    help="🌊 Simulate, test, and replay robotic pool cleaners.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

BANNER = (
    "[bold cyan]🌊 ZimaBlue[/bold cyan] "
    "[dim]· simulate, test, and replay robotic pool cleaners[/dim]"
)


# Rich parses square brackets as style tags, so the extra is escaped here once
# rather than at each use -- an unescaped hint renders as "pip install
# 'zimablue'", which is exactly the wrong advice.
_VIZ_MISSING = (
    "[yellow]matplotlib not installed; skipping images (pip install 'zimablue\\[viz]')[/yellow]"
)
_VIZ_HINT = "pip install 'zimablue\\[viz]' -- or render headless with --gif out.gif"


def _guard_viz() -> None:
    """Turn a missing optional dependency into a clean CLI error.

    The library raises a perfectly good ModuleNotFoundError; what a CLI user
    should see is one line telling them what to install, not a traceback
    through our rendering internals.
    """
    from zimablue.replay import require_matplotlib

    try:
        require_matplotlib()
    except ModuleNotFoundError as exc:
        _fail(str(exc), "or install everything: pip install 'zimablue[dev]'")


def _fail(message: str, hint: str | None = None) -> None:
    """Print an error and exit.

    Both strings are escaped: they carry exception text and install commands,
    and Rich reads square brackets as style tags. Unescaped, the advice
    "pip install 'zimablue[viz]'" renders as "pip install 'zimablue'" -- which
    is precisely what the user already did.
    """
    console.print(f"[bold red]error[/bold red] {escape(message)}")
    if hint:
        console.print(f"[dim]hint:[/dim] {escape(hint)}")
    raise typer.Exit(code=1)


def _metrics_table(metrics: Any, title: str = "results") -> Table:
    table = Table(title=title, title_style="bold cyan", show_header=False, box=None, pad_edge=False)
    table.add_column("metric", style="dim", width=20)
    table.add_column("value", style="bold")

    def row(label: str, value: str, style: str = "") -> None:
        table.add_row(label, f"[{style}]{value}[/{style}]" if style else value)

    row("coverage", f"{metrics.coverage * 100:.1f} %", "cyan")
    row("wall coverage", f"{metrics.wall_coverage * 100:.1f} %", "cyan")
    row("dirt removed", f"{metrics.dirt_removed_fraction * 100:.1f} %", "green")
    row(
        "", f"{metrics.dirt_removed:.0f} g of {metrics.dirt_removed + metrics.remaining_dirt:.0f} g"
    )
    row("uniformity", f"{metrics.cleaning_uniformity * 100:.1f} %")
    row("revisits", f"{metrics.revisits:.2f} extra passes/cell")
    row("distance", f"{metrics.distance_traveled:.1f} m")
    row("runtime", f"{metrics.runtime / 60:.1f} min")
    row("energy", f"{metrics.energy_consumed:.1f} Wh", "yellow")
    row("battery left", f"{metrics.battery_remaining * 100:.0f} %", "yellow")
    row("collisions", str(metrics.collisions))
    row("stuck events", str(metrics.stuck_events))
    row("termination", metrics.termination)
    return table


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"zimablue {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """🌊 ZimaBlue."""


# ----------------------------------------------------------------------
@app.command()
def demo(
    pool: Annotated[str, typer.Option(help="Pool preset.")] = "kidney",
    robot: Annotated[str, typer.Option(help="Cleaner preset.")] = "tracked",
    dirt: Annotated[str, typer.Option(help="Dirt preset.")] = "autumn",
    minutes: Annotated[float, typer.Option(help="Simulated duration.")] = 20.0,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    out: Annotated[Path, typer.Option(help="Where to write outputs.")] = Path("runs"),
    watch: Annotated[bool, typer.Option(help="Open the interactive replay window.")] = True,
    gif: Annotated[bool, typer.Option(help="Also render an animated GIF.")] = False,
) -> None:
    """Run a complete example: dirty pool in, cleaned pool and a replay out."""
    from zimablue.simulation import Simulation

    console.print(Panel(BANNER, border_style="cyan"))
    console.print(
        f"[dim]pool[/dim] {pool}   [dim]robot[/dim] {robot}   [dim]dirt[/dim] {dirt}   "
        f"[dim]seed[/dim] {seed}   [dim]duration[/dim] {minutes:g} min\n"
    )

    try:
        sim = Simulation(pool=pool, robot=robot, dirt=dirt, seed=seed, scenario_name=f"demo_{pool}")
    except KeyError as exc:
        _fail(str(exc).strip("\"'"))

    console.print(f"[dim]pool[/dim]  {sim.pool}")
    console.print(f"[dim]robot[/dim] {sim.robot.describe()}")
    console.print(f"[dim]dirt[/dim]  {sim.world.dirt.initial_mass:.0f} g to remove\n")

    result = _run_with_progress(sim, minutes * 60.0)
    console.print(_metrics_table(result.metrics, "how did it do?"))

    out.mkdir(parents=True, exist_ok=True)
    recording_path = result.save(out / f"demo_{pool}_{seed}.zbr")
    console.print(
        f"\n[green]recorded[/green] {recording_path} ({recording_path.stat().st_size / 1e6:.1f} MB)"
    )

    summary_path = out / f"demo_{pool}_{seed}_summary.png"
    try:
        from zimablue.replay import export_summary

        export_summary(result.recording, summary_path)
        console.print(f"[green]summary [/green] {summary_path}")
    except ImportError:
        console.print(_VIZ_MISSING)
        return

    if gif:
        from zimablue.replay import export_movie

        gif_path = out / f"demo_{pool}_{seed}.gif"
        with console.status("rendering GIF..."):
            export_movie(result.recording, gif_path, speed=90.0)
        console.print(f"[green]gif     [/green] {gif_path}")

    if watch:
        _watch(result.recording)
    else:
        console.print(f"\n[dim]watch it:[/dim] zimablue replay {recording_path}")


# ----------------------------------------------------------------------
@app.command()
def run(
    scenario_file: Annotated[
        Path, typer.Argument(help="Scenario YAML file, or a built-in name like 'kidney'.")
    ],
    seed: Annotated[int | None, typer.Option(help="Override the scenario seed.")] = None,
    record: Annotated[Path | None, typer.Option(help="Write a .zbr recording here.")] = None,
    minutes: Annotated[float | None, typer.Option(help="Override the duration.")] = None,
    summary: Annotated[Path | None, typer.Option(help="Write a summary PNG here.")] = None,
    quiet: Annotated[bool, typer.Option(help="Print metrics only.")] = False,
) -> None:
    """Run a scenario from a YAML file."""
    from zimablue.scenarios import load_scenario

    try:
        scenario = load_scenario(scenario_file)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    if minutes is not None:
        scenario.duration = minutes * 60.0
    if not quiet:
        console.print(Panel(BANNER, border_style="cyan"))
        console.print(f"[bold]{scenario.describe()}[/bold]\n")

    sim = scenario.simulation(seed, record=record is not None)
    result = (
        _run_with_progress(sim, scenario.duration)
        if not quiet
        else sim.run(seconds=scenario.duration)
    )
    console.print(_metrics_table(result.metrics, scenario.name))

    if record is not None:
        path = result.save(record)
        console.print(f"\n[green]recorded[/green] {path}")
    if summary is not None:
        from zimablue.replay import export_summary

        if result.recording is None:
            _fail("--summary needs a recording", "add --record runs/out.zbr")
        else:
            export_summary(result.recording, summary)
        console.print(f"[green]summary [/green] {summary}")


# ----------------------------------------------------------------------
@app.command()
def replay(
    recording: Annotated[Path, typer.Argument(help="A .zbr recording.")],
    speed: Annotated[float, typer.Option(help="Initial playback speed.")] = 8.0,
    gif: Annotated[Path | None, typer.Option(help="Render to GIF/MP4 instead of watching.")] = None,
    summary: Annotated[Path | None, typer.Option(help="Write a summary PNG instead.")] = None,
    frames: Annotated[Path | None, typer.Option(help="Write still frames to a directory.")] = None,
    sensors: Annotated[bool, typer.Option(help="Draw sonar rays.")] = True,
    three_d: Annotated[
        bool,
        typer.Option("--3d/--2d", help="Render the pool as a 3D basin (file output only)."),
    ] = False,
) -> None:
    """Replay a recorded run -- interactively, or to a file."""
    from zimablue.recording import Recording

    try:
        rec = Recording.load(recording)
    except (FileNotFoundError, ValueError) as exc:
        _fail(
            str(exc), "make one with: zimablue demo  or  zimablue run <scenario> --record out.zbr"
        )

    if gif is not None or summary is not None or frames is not None or not three_d:
        _guard_viz()

    if gif is not None:
        if three_d:
            from zimablue.replay import export_3d_movie

            with console.status(f"rendering {gif} in 3D..."):
                export_3d_movie(rec, gif, speed=max(speed * 10, 60.0))
        else:
            from zimablue.replay import export_movie

            with console.status(f"rendering {gif}..."):
                export_movie(rec, gif, speed=max(speed * 10, 60.0), show_sensors=sensors)
        console.print(f"[green]wrote[/green] {gif}")
        return
    if summary is not None:
        if three_d:
            from zimablue.replay import export_3d_frames

            export_3d_frames(rec, summary, count=4)
        else:
            from zimablue.replay import export_summary

            export_summary(rec, summary)
        console.print(f"[green]wrote[/green] {summary}")
        return
    if three_d:
        _fail(
            "3D replay renders to a file, not an interactive window.",
            "try: zimablue replay run.zbr --3d --gif out.gif",
        )
    if frames is not None:
        from zimablue.replay import export_frames

        written = export_frames(rec, frames)
        console.print(f"[green]wrote[/green] {len(written)} frames to {frames}")
        return

    _watch(rec, speed=speed, sensors=sensors)


# ----------------------------------------------------------------------
@app.command()
def batch(
    scenario_file: Annotated[
        Path, typer.Argument(help="Scenario YAML file, or a built-in name like 'kidney'.")
    ],
    episodes: Annotated[int, typer.Option(help="How many seeded episodes.")] = 20,
    record_dir: Annotated[Path | None, typer.Option(help="Keep every episode's .zbr here.")] = None,
    out: Annotated[Path | None, typer.Option(help="Write aggregate JSON here.")] = None,
    csv: Annotated[Path | None, typer.Option(help="Write per-episode CSV here.")] = None,
    minutes: Annotated[float | None, typer.Option(help="Override the duration.")] = None,
) -> None:
    """Run many seeded episodes and report aggregate statistics."""
    from zimablue.batch import run_batch
    from zimablue.scenarios import load_scenario

    try:
        scenario = load_scenario(scenario_file)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    console.print(Panel(BANNER, border_style="cyan"))
    console.print(f"[bold]{scenario.describe()}[/bold]")
    console.print(
        f"[dim]{episodes} episodes, seeds {scenario.seed}..{scenario.seed + episodes - 1}[/dim]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("episodes", total=episodes)

        def tick(index: int, episode: Any) -> None:
            progress.update(
                task,
                advance=1,
                description=f"seed {episode.seed}  cov {episode.metrics.coverage * 100:.0f}%",
            )

        result = run_batch(scenario, episodes=episodes, record_dir=record_dir, on_episode=tick)

    console.print()
    console.print(Panel(result.summary(), title="aggregate", border_style="cyan", expand=False))

    worst = result.worst("coverage", 3)
    if worst:
        table = Table(title="worst episodes", title_style="bold red", box=None)
        table.add_column("seed")
        table.add_column("coverage")
        table.add_column("dirt removed")
        table.add_column("termination")
        for episode in worst:
            table.add_row(
                str(episode.seed),
                f"{episode.metrics.coverage * 100:.1f} %",
                f"{episode.metrics.dirt_removed_fraction * 100:.1f} %",
                episode.metrics.termination,
            )
        console.print()
        console.print(table)
        console.print(
            f"\n[dim]reproduce the worst one:[/dim] "
            f"zimablue run {scenario_file} --seed {worst[0].seed} --record runs/worst.zbr"
        )

    if out is not None:
        console.print(f"\n[green]wrote[/green] {result.save(out)}")
    if csv is not None:
        console.print(f"[green]wrote[/green] {result.to_csv(csv)}")


# ----------------------------------------------------------------------
@app.command(name="inspect")
def inspect_recording(
    recording: Annotated[Path, typer.Argument(help="A .zbr recording.")],
    channels: Annotated[bool, typer.Option(help="List every recorded channel.")] = False,
    events: Annotated[bool, typer.Option(help="List recorded events.")] = False,
) -> None:
    """Show what is inside a recording, without replaying it."""
    from zimablue.recording import Recording

    try:
        rec = Recording.load(recording)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    console.print(Panel(rec.describe(), title=str(recording), border_style="cyan", expand=False))

    if rec.metrics:
        from zimablue.metrics import Metrics

        console.print(_metrics_table(Metrics.from_dict(rec.metrics), "recorded metrics"))
    if channels:
        console.print("\n[bold]channels[/bold]")
        for name in rec.channels:
            console.print(f"  {name}")
    if events:
        from collections import Counter

        counts = Counter(e["kind"] for e in rec.events)
        console.print("\n[bold]events[/bold]")
        for kind, count in counts.most_common():
            console.print(f"  {kind:20s} {count}")


# ----------------------------------------------------------------------
@app.command(name="list")
def list_presets() -> None:
    """List every available preset."""
    from zimablue.backends.base import BACKENDS
    from zimablue.controllers.base import CONTROLLERS
    from zimablue.dirt import DIRT_PRESETS
    from zimablue.pool import POOL_PRESETS
    from zimablue.robot import ROBOT_PRESETS

    console.print(Panel(BANNER, border_style="cyan"))
    table = Table(box=None)
    table.add_column("kind", style="dim")
    table.add_column("presets", style="cyan")
    for label, registry in (
        ("pools", POOL_PRESETS),
        ("robots", ROBOT_PRESETS),
        ("dirt", DIRT_PRESETS),
        ("controllers", CONTROLLERS),
        ("backends", BACKENDS),
    ):
        table.add_row(label, ", ".join(registry.names()) or "[dim](none)[/dim]")
    console.print(table)


# ----------------------------------------------------------------------
def _run_with_progress(sim: Any, duration: float) -> Any:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan"),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("simulating", total=duration)

        def report(elapsed: float, total: float) -> None:
            progress.update(task, completed=min(elapsed, total))

        result = sim.run(seconds=duration, progress=report)
        progress.update(task, completed=duration)
    return result


def _watch(recording: Any, *, speed: float = 8.0, sensors: bool = True) -> None:
    """Open the interactive player, explaining clearly if there is no display."""
    try:
        import matplotlib
    except ImportError:
        _fail(
            "matplotlib is not installed",
            _VIZ_HINT,
        )
    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        console.print(
            "[yellow]no interactive display available[/yellow] "
            "(matplotlib is using the Agg backend)."
        )
        console.print("[dim]render it instead:[/dim] zimablue replay <file.zbr> --gif run.gif")
        return
    from zimablue.replay import ReplayPlayer

    console.print(
        "\n[dim]space[/dim] pause  [dim]←/→[/dim] step  [dim]↑/↓[/dim] speed  "
        "[dim]r[/dim] restart  [dim]s[/dim] snapshot  [dim]q[/dim] quit"
    )
    ReplayPlayer(recording, speed=speed, show_sensors=sensors).show()


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
