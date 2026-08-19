"""The ``zimablue`` command line.

Verbs: ``demo``, ``run``, ``replay``, ``trace``, ``batch``, ``compare``,
``inspect``, ``list``.

Errors are meant to be actionable -- an unknown preset lists the valid ones, a
missing recording says where to make one -- because the most common CLI failure
is a typo, and a stack trace is a poor answer to a typo.
"""

from __future__ import annotations

import json
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
    row("", f"{getattr(metrics, 'grams_per_wh', 0.0):.1f} g captured per Wh")
    row("battery left", f"{metrics.battery_remaining * 100:.0f} %", "yellow")
    row("collisions", str(metrics.collisions))
    row("stuck events", str(metrics.stuck_events))
    skimmed = getattr(metrics, "debris_skimmed", 0)
    debris_total = (
        getattr(metrics, "debris_collected", 0) + skimmed + getattr(metrics, "debris_remaining", 0)
    )
    if debris_total:
        oversize = getattr(metrics, "debris_oversize", 0)
        row("debris", f"{metrics.debris_collected} of {debris_total} collected")
        if skimmed:
            row("", f"{skimmed} taken by the skimmer, not the robot")
        if oversize:
            # The number that reframes the dirt figure above it: these items
            # were never collectable by this robot, whatever it did.
            row("", f"{oversize} too big for the intake", "yellow")
            row(
                "dirt ceiling",
                f"{getattr(metrics, 'dirt_ceiling', 1.0) * 100:.1f} %",
                "yellow",
            )
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
    html: Annotated[
        Path | None,
        typer.Option(help="Write a self-contained HTML player instead. Needs no matplotlib."),
    ] = None,
    sensors: Annotated[bool, typer.Option(help="Draw sonar rays.")] = True,
    three_d: Annotated[
        bool,
        typer.Option("--3d/--2d", help="Render the pool as a 3D basin (file output only)."),
    ] = False,
    dirt_cam: Annotated[
        bool,
        typer.Option("--dirtcam", help="Render from the cleaner's bumper (file output only)."),
    ] = False,
    chase_cam: Annotated[
        bool,
        typer.Option("--chase", help="Render from behind the cleaner (file output only)."),
    ] = False,
    map_panel: Annotated[
        bool,
        typer.Option("--map/--no-map", help="Keep the top-down panel beside the dirt cam."),
    ] = True,
) -> None:
    """Replay a recorded run -- interactively, or to a file."""
    from zimablue.recording import Recording

    try:
        rec = Recording.load(recording)
    except (FileNotFoundError, ValueError) as exc:
        _fail(
            str(exc), "make one with: zimablue demo  or  zimablue run <scenario> --record out.zbr"
        )

    if html is not None:
        from zimablue.replay import export_web_player

        export_web_player(rec, html)
        console.print(f"[green]wrote[/green] {html}")
        console.print("[dim]open it in any browser; it works from disk, offline[/dim]")
        return

    # Four cameras and counting, so check them as a set rather than pairwise:
    # the ad-hoc "if a and b" grew a hole the moment a third one arrived.
    cameras = {"--3d": three_d, "--dirtcam": dirt_cam, "--chase": chase_cam}
    chosen = [flag for flag, on in cameras.items() if on]
    if len(chosen) > 1:
        _fail(
            f"{' and '.join(chosen)} are different cameras.",
            "pick one: --3d looks down at the basin, --chase follows from behind, "
            "--dirtcam looks out from the robot itself",
        )

    if gif is not None or summary is not None or frames is not None or not three_d:
        _guard_viz()

    if dirt_cam or chase_cam:
        which = "--dirtcam" if dirt_cam else "--chase"
        where = "from the bumper" if dirt_cam else "from behind"
        if gif is None and summary is None:
            _fail(
                f"{which} renders to a file, not an interactive window.",
                f"try: zimablue replay run.zbr {which} --gif out.gif",
            )
        if gif is not None:
            with console.status(f"rendering {gif} {where}..."):
                if dirt_cam:
                    from zimablue.replay import export_dirtcam

                    export_dirtcam(rec, gif, speed=max(speed * 10, 60.0), with_map=map_panel)
                else:
                    from zimablue.replay import export_chasecam

                    export_chasecam(rec, gif, speed=max(speed * 10, 60.0))
            console.print(f"[green]wrote[/green] {gif}")
        if summary is not None:
            if dirt_cam:
                from zimablue.replay import export_dirtcam_frames

                export_dirtcam_frames(rec, summary)
            else:
                from zimablue.replay import export_chasecam_frames

                export_chasecam_frames(rec, summary)
            console.print(f"[green]wrote[/green] {summary}")
        return

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
def trace(
    picture: Annotated[Path, typer.Argument(help="A photo of a pool.")],
    width: Annotated[float | None, typer.Option(help="The pool's real width in metres.")] = None,
    metres_per_pixel: Annotated[
        float | None, typer.Option("--mpp", help="Ground resolution, for a top-down image.")
    ] = None,
    sample: Annotated[
        str | None, typer.Option(help="'x,y' pixel inside the water. Use this for a photo.")
    ] = None,
    depth: Annotated[float, typer.Option(help="Flat depth in metres -- a photo cannot say.")] = 1.5,
    name: Annotated[str, typer.Option(help="Name for the traced pool.")] = "traced",
    check: Annotated[
        Path | None, typer.Option(help="Write an overlay of what was found. Look at it.")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Write the pool as JSON.")] = None,
) -> None:
    """Trace a pool out of a photograph.

    Needs a scale: a photograph does not carry one. For an oblique shot, use
    the Python API's corners= to correct the perspective as well.
    """
    from zimablue.imaging import require_pillow, trace_pool

    try:
        require_pillow()
    except ModuleNotFoundError as exc:
        _fail(str(exc), "or install everything: pip install 'zimablue[dev]'")

    if (width is None) == (metres_per_pixel is None):
        _fail(
            "give exactly one of --width or --mpp.",
            "a photograph does not contain its own scale, so there is no default",
        )

    seed = None
    if sample is not None:
        try:
            sx, sy = (int(part) for part in sample.replace(" ", "").split(","))
        except ValueError:
            _fail(f"could not read --sample {sample!r}", "it should look like --sample 640,410")
        seed = (sx, sy)

    try:
        traced = trace_pool(picture, sample=seed, width=width, metres_per_pixel=metres_per_pixel)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    console.print(traced.summary())
    pool = traced.pool(depth, name=name)

    if check is not None:
        _guard_viz()
        traced.overlay(check)
        console.print(f"[green]wrote[/green] {check}")
    else:
        console.print(
            "[yellow]no --check given[/yellow]; segmenting a photo is a guess, "
            "and the overlay is how you catch a wrong one"
        )

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(pool.to_dict(), indent=2))
        console.print(f"[green]wrote[/green] {out}")


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
@app.command()
def bench(
    out: Annotated[Path, typer.Option(help="Directory for the JSON, CSV and markdown.")] = Path(
        "runs/bench"
    ),
    jobs: Annotated[int, typer.Option(help="Worker processes. Runs are independent.")] = 1,
    quick: Annotated[
        bool,
        typer.Option(help="The smoke tier: proves the pipeline in a minute, means nothing."),
    ] = False,
) -> None:
    """Run the frozen benchmark suite and write the leaderboard.

    Nothing about the suite is configurable -- that is what makes two results
    comparable. Use 'zimablue compare' for a comparison on your own terms.
    """
    from zimablue.bench import BENCH_QUICK, BENCH_V1, run_bench

    definition = BENCH_QUICK if quick else BENCH_V1
    console.print(Panel(BANNER, border_style="cyan"))
    console.print(f"[bold]{definition.describe()}[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("benchmarking", total=definition.runs)

        def tick(trial: Any) -> None:
            progress.update(task, advance=1, description=f"{trial.planner} on {trial.pool}")

        result = run_bench(definition, jobs=jobs, on_result=tick)

    console.print()
    console.print(_comparison_table(result.comparison))
    for kind, path in result.save(out).items():
        console.print(f"[green]wrote[/green] {path}  [dim]({kind})[/dim]")


# ----------------------------------------------------------------------
@app.command()
def compare(
    planners: Annotated[
        list[str] | None,
        typer.Argument(
            help="Planner entries, like 'bsa' or 'morse@odometry'. Default: everything."
        ),
    ] = None,
    pool: Annotated[
        list[str] | None, typer.Option("--pool", help="Pool preset. Repeat for several.")
    ] = None,
    seeds: Annotated[int, typer.Option(help="Seeds per planner and pool, starting at 1.")] = 1,
    minutes: Annotated[float, typer.Option(help="Simulated duration per run.")] = 20.0,
    dirt: Annotated[str, typer.Option(help="Dirt preset.")] = "autumn",
    robot: Annotated[str, typer.Option(help="Cleaner preset.")] = "tracked",
    jobs: Annotated[int, typer.Option(help="Worker processes. Trials are independent.")] = 1,
    localisation: Annotated[
        str,
        typer.Option(help="How offline planners are followed: odometry, truth, or both."),
    ] = "odometry",
    fleet: Annotated[
        int, typer.Option(help="Compare teams of this many robots instead of single cleaners.")
    ] = 0,
    csv: Annotated[Path | None, typer.Option(help="Write per-trial CSV here.")] = None,
    matrix: Annotated[Path | None, typer.Option(help="Write the matrix plot PNG here.")] = None,
) -> None:
    """Run the planner leaderboard: every entry on every pool, measured on every axis."""
    from zimablue.planners import compare as harness

    if localisation not in ("odometry", "truth", "both"):
        _fail(
            f"unknown localisation {localisation!r}",
            "pick odometry, truth, or both",
        )
    entries = tuple(planners) if planners else None
    if fleet > 0:
        if entries is None:
            entries = harness.FLEET_ENTRIES
        _check_fleet_entries(entries)
    else:
        _check_entries(entries or harness.default_entries(localisation=localisation))
    pools = tuple(pool) if pool else ("rectangular",)
    seed_tuple = tuple(range(1, seeds + 1))

    total = len(entries or harness.default_entries(localisation=localisation))
    total *= len(pools) * len(seed_tuple)
    console.print(Panel(BANNER, border_style="cyan"))
    console.print(
        f"[dim]pools[/dim] {', '.join(pools)}   [dim]dirt[/dim] {dirt}   "
        f"[dim]duration[/dim] {minutes:g} min   [dim]runs[/dim] {total}\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("comparing", total=total)

        def tick(trial: Any) -> None:
            progress.update(task, advance=1, description=f"{trial.planner} on {trial.pool}")

        if fleet > 0:
            result = harness.compare_fleets(
                entries or harness.FLEET_ENTRIES,
                robots=fleet,
                pools=pools,
                seeds=seed_tuple,
                minutes=minutes,
                dirt=dirt,
                jobs=jobs,
                on_result=tick,
            )
        else:
            result = harness.compare(
                entries,
                pools=pools,
                seeds=seed_tuple,
                minutes=minutes,
                dirt=dirt,
                robot=robot,
                jobs=jobs,
                localisation=localisation,
                on_result=tick,
            )

    console.print()
    console.print(_comparison_table(result))
    if len(pools) > 1 or seeds > 1:
        console.print(f"[dim]median over {len(pools)} pool(s), {len(result.trials)} runs[/dim]")

    if csv is not None:
        csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(csv)
        console.print(f"\n[green]wrote[/green] {csv}")
    if matrix is not None:
        _guard_viz()
        from zimablue.planners.plots import plot_comparison

        figure = plot_comparison(result)
        matrix.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(matrix, dpi=150, bbox_inches="tight")
        console.print(f"[green]wrote[/green] {matrix}")


def _check_entries(entries: tuple[str, ...]) -> None:
    """Reject a typo before it costs twenty simulated minutes."""
    from zimablue.controllers.base import CONTROLLERS
    from zimablue.planners import PLANNERS

    for entry in entries:
        if "@" in entry:
            name, mode = entry.split("@", 1)
            if name not in PLANNERS:
                _fail(
                    f"unknown planner {name!r}",
                    f"offline planners: {', '.join(PLANNERS.names())}",
                )
            if mode not in ("truth", "odometry"):
                _fail(f"unknown localisation {mode!r} in {entry!r}", "use @truth or @odometry")
        elif entry not in CONTROLLERS:
            _fail(
                f"unknown entry {entry!r}",
                f"controllers: {', '.join(CONTROLLERS.names())}; "
                f"offline planners take @truth or @odometry: {', '.join(PLANNERS.names())}",
            )


def _check_fleet_entries(entries: tuple[str, ...]) -> None:
    from zimablue.controllers.base import CONTROLLERS
    from zimablue.planners import PARTITIONS, PLANNERS

    for entry in entries:
        if entry in ("mstc", "mstc_nobt"):
            continue
        if "+" in entry:
            method, planner = entry.split("+", 1)
            if method not in PARTITIONS:
                _fail(
                    f"unknown partition {method!r} in {entry!r}",
                    f"partitions: {', '.join(PARTITIONS.names())}",
                )
            if planner not in PLANNERS:
                _fail(
                    f"unknown planner {planner!r} in {entry!r}",
                    f"offline planners: {', '.join(PLANNERS.names())}",
                )
        elif entry not in CONTROLLERS:
            _fail(
                f"unknown entry {entry!r}",
                f"controllers: {', '.join(CONTROLLERS.names())}, mstc, mstc_nobt, "
                "or partition+planner like darp+sweep_optimal",
            )


def _comparison_table(comparison: Any) -> Table:
    """The measurements as a Rich table, best value in each column marked."""
    import numpy as np

    table = Table(title=comparison.label, title_style="bold cyan", box=None)
    table.add_column("planner", style="bold")
    for dim in comparison.dimensions:
        table.add_column(dim.label, justify="right")
    raw = {
        planner: [comparison.score(planner, d.key) for d in comparison.dimensions]
        for planner in comparison.planners
    }
    winners = []
    for j, dim in enumerate(comparison.dimensions):
        column = [v for p in comparison.planners if np.isfinite(v := raw[p][j])]
        winners.append((max(column) if dim.better > 0 else min(column)) if column else None)
    for planner in comparison.planners:
        cells = [planner]
        for j, dim in enumerate(comparison.dimensions):
            value = raw[planner][j]
            text = dim.format(value)
            best = winners[j]
            if best is not None and np.isfinite(value) and np.isclose(value, best):
                text = f"[bold cyan]*{text}[/bold cyan]"
            cells.append(text)
        table.add_row(*cells)
    return table


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

    if rec.metrics and rec.has_ground_truth:
        from zimablue.metrics import Metrics

        console.print(_metrics_table(Metrics.from_dict(rec.metrics), "recorded metrics"))
    elif rec.metrics:
        # A recording made on a robot carries a different, shorter set of
        # metrics. Pouring it into Metrics.from_dict fills the missing keys
        # with zeros, which reads as a controller that drove nowhere and
        # cleaned nothing rather than as a run nobody measured.
        table = Table(title="recorded metrics", box=None, show_header=False)
        for key, value in sorted(rec.metrics.items()):
            table.add_row(key, f"{value:g}" if isinstance(value, int | float) else str(value))
        console.print(table)
        console.print(
            "[dim]No ground truth in this recording, so there is no coverage or "
            "dirt-removed figure -- both need the true pose and the true dirt "
            "field. See docs/hardware.md.[/dim]"
        )
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
    from zimablue.planners import PARTITIONS, PLANNERS
    from zimablue.pool import POOL_PRESETS
    from zimablue.robot import DESIGNS, ROBOT_PRESETS

    console.print(Panel(BANNER, border_style="cyan"))
    table = Table(box=None)
    table.add_column("kind", style="dim")
    table.add_column("presets", style="cyan")
    for label, registry in (
        ("pools", POOL_PRESETS),
        ("robots", ROBOT_PRESETS),
        ("designs", DESIGNS),
        ("dirt", DIRT_PRESETS),
        ("controllers", CONTROLLERS),
        ("planners", PLANNERS),
        ("partitions", PARTITIONS),
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
