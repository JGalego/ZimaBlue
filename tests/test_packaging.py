"""What a `pip install zimablue` actually gets.

These are the things that only break for users, never for developers: the repo
has a `scenarios/` directory and a full dev environment, so a wheel that ships
neither still passes every other test in this suite.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import zimablue as zb
from zimablue.scenarios import bundled_scenarios, load_scenario, resolve_scenario

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomllib entered the standard library in 3.11; 3.10 is our floor.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent

# hatch_build.py lives at the repository root, outside the package.
sys.path.insert(0, str(ROOT))


def test_version_has_one_source():
    """pyproject reads the version from the module, so they cannot drift."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" in config["project"].get("dynamic", []), (
        "pyproject should take the version from src/zimablue/_version.py"
    )
    assert config["tool"]["hatch"]["version"]["path"] == "src/zimablue/_version.py"


def test_scenarios_are_declared_as_package_data():
    """Without this, every documented `zimablue run` needs a git clone."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert include["scenarios"] == "zimablue/data/scenarios"


def test_repo_scenarios_all_load():
    for path in sorted((ROOT / "scenarios").glob("*.yaml")):
        scenario = load_scenario(path)
        assert scenario.seed >= 0
        assert scenario.duration > 0


def test_a_bare_name_resolves_when_scenarios_are_bundled():
    bundled = bundled_scenarios()
    if not bundled:
        pytest.skip("running from a source tree without the built data directory")
    name = next(iter(bundled))
    assert resolve_scenario(name).exists()
    assert load_scenario(name).seed >= 0


def test_a_local_path_wins_over_a_bundled_name(tmp_path):
    local = tmp_path / "kidney.yaml"
    local.write_text("name: local\npool:\n  preset: rectangular\nseed: 7\n")
    assert resolve_scenario(local) == local


def test_unknown_scenario_names_the_alternatives():
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_scenario("definitely-not-a-scenario")
    message = str(excinfo.value)
    assert "definitely-not-a-scenario" in message
    assert "built-in names" in message


def test_importing_zimablue_does_not_import_matplotlib():
    """The core install has no matplotlib, so importing it at module scope
    would make `import zimablue` fail for everyone who skipped the extra."""
    import subprocess
    import sys

    code = "import zimablue, sys; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_public_api_is_importable_and_complete():
    missing = [name for name in zb.__all__ if not hasattr(zb, name)]
    assert not missing, f"__all__ lists names that do not exist: {missing}"


def test_viz_hint_names_the_extra():
    from zimablue.replay import VIZ_HINT

    assert "zimablue[viz]" in VIZ_HINT


def test_image_hint_names_the_extra():
    from zimablue.imaging import IMAGE_HINT

    assert "zimablue[image]" in IMAGE_HINT


def test_ml_hints_name_their_extras():
    from zimablue.segment import ONNX_HINT

    assert "zimablue[ml]" in ONNX_HINT

    # zimablue.rl raises on import without gymnasium, so read the constant out
    # of the source rather than importing the package to find out.
    text = (ROOT / "src" / "zimablue" / "rl" / "__init__.py").read_text()
    assert "zimablue[rl]" in text


def test_importing_zimablue_does_not_import_a_model_runtime():
    """The core install has no machine learning in it and should prove it."""
    import subprocess
    import sys

    code = "import zimablue, sys; print(sorted({'onnxruntime', 'gymnasium'} & set(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_importing_zimablue_does_not_import_pillow():
    """Same contract as matplotlib: the optional extras stay optional.

    ``zimablue.imaging`` is imported at package import, so its Pillow import
    has to stay inside the functions that read a file.
    """
    import subprocess
    import sys

    code = "import zimablue, sys; print('PIL' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_every_readme_anchor_points_at_a_real_heading():
    """A table of contents rots the first time a heading is renamed.

    GitHub silently scrolls nowhere for a dead anchor, so nothing complains
    except the reader.
    """
    text = (ROOT / "README.md").read_text()

    def slug(heading: str) -> str:
        cleaned = re.sub(r"[^\w\s-]", "", heading.lower().strip())
        return re.sub(r"\s+", "-", cleaned)

    headings = {slug(m.group(2)) for m in re.finditer(r"^(#{2,3})\s+(.*)$", text, re.M)}
    dead = [a for a in re.findall(r"\]\(#([^)]+)\)", text) if a not in headings]
    assert not dead, f"README links to headings that do not exist: {dead}"


def test_the_readme_lists_every_replay_camera():
    """Four cameras now. A view nobody can find is a view nobody uses."""
    text = (ROOT / "README.md").read_text()
    for flag in ("--chase", "--dirtcam", "--3d"):
        assert flag in text, f"{flag} is not mentioned in the README"


def test_the_hardware_module_needs_nothing_installed():
    """It runs on a robot. A robot is the last place you want a dependency tree.

    Also the last place you want matplotlib: importing zimablue.hardware must
    not drag in a plotting stack, an ONNX runtime or Gymnasium.
    """
    import subprocess
    import sys

    code = (
        "import zimablue.hardware, sys; "
        "print(sorted({'matplotlib', 'onnxruntime', 'gymnasium', 'PIL'} & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_the_hardware_runtime_and_the_simulator_share_one_frame_builder():
    """Two writers of the same format is two formats sharing a file extension.

    The columns a ``.zbr`` carries are not a detail: replay, metrics and every
    downstream reader are written against them, so a recording produced on a
    robot has to be laid out exactly like one produced by the backend.
    """
    simulation = (ROOT / "src" / "zimablue" / "simulation.py").read_text()
    runtime = (ROOT / "src" / "zimablue" / "hardware" / "runtime.py").read_text()
    assert "build_frame(" in simulation and "build_frame(" in runtime
    assert '"cmd_left": command.left' not in runtime, (
        "the hardware runtime is building frames by hand again; it must use "
        "recording.build_frame or the two layouts will drift"
    )


def test_readme_urls_are_absolute_in_package_metadata():
    """PyPI renders the description with no repository behind it.

    Relative paths that work on GitHub are dead there: images fail silently
    and links 404. The build hook rewrites them; this checks it did.
    """
    from hatch_build import absolutise

    text = absolutise((ROOT / "README.md").read_text(), ref="v9.9.9")

    images = re.findall(r'<img[^>]+src="([^"]+)"', text)
    assert images, "expected the README to contain images"
    relative = [u for u in images if not u.startswith("http")]
    assert not relative, f"relative image sources survive: {relative}"

    links = re.findall(r"\]\(([^)]+)\)", text)
    dangling = [u for u in links if not u.startswith(("http", "mailto:", "#"))]
    assert not dangling, f"relative links survive: {dangling}"

    assert "/v9.9.9/" in text, "the ref should be substituted into the URLs"


def test_the_readme_has_one_visual_hero():
    text = (ROOT / "README.md").read_text()
    assert "logo-animated.svg" not in text
    assert "docs/assets/replay.gif" in text


def test_every_referenced_asset_exists():
    """A rewritten URL still 404s if the file was never committed."""
    from hatch_build import absolutise

    text = absolutise((ROOT / "README.md").read_text())
    prefix = "https://raw.githubusercontent.com/JGalego/ZimaBlue/main/"
    for url in re.findall(r'<img[^>]+src="([^"]+)"', text):
        if url.startswith(prefix):
            asset = ROOT / url[len(prefix) :]
            assert asset.exists(), f"README points at a missing file: {asset}"


def test_description_renders_the_way_pypi_renders_it():
    """`twine check` only asks whether it renders, not whether anything
    survives the sanitiser. This asks the second question."""
    renderer = pytest.importorskip("readme_renderer.markdown")
    # readme_renderer without its [md] extra imports fine and then renders
    # *nothing* -- render() returns None for every markdown document. That is
    # a broken environment, not a broken README, so skip rather than fail.
    pytest.importorskip("cmarkgfm")
    from hatch_build import absolutise

    html = renderer.render(absolutise((ROOT / "README.md").read_text()))
    assert html is not None, "PyPI would reject this description"
    assert html.count("<img") >= 5, "images were stripped by the sanitiser"
    assert "<table" in html and "<pre" in html


def test_the_release_guard_reads_the_version_it_will_publish():
    """The tag check has one job and got it wrong the first time it ran.

    It parsed ``_version.py`` with the regex ``"(.+)"``, which matches the
    module docstring's opening triple quote long before it reaches
    ``__version__`` -- so it compared the tag against a single ``"`` and
    failed. Nothing caught it because the step only runs on a tag push, and
    0.1.0 was published through workflow_dispatch.

    It reads the built wheel's filename now. That is also the better source:
    the filename is what lands on PyPI, and it cannot disagree with the
    artefact the way a re-parse of the source can.
    """
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    guard = workflow[workflow.index("Tag must match the package version") :]
    guard = guard[: guard.index("- name:", 10)]

    assert "dist" in guard and ".whl" in guard, "the guard should read the built wheel"
    assert "_version.py" not in guard.split('"""')[0] or "regex" in guard, (
        "re-parsing the source is what broke; if it comes back, it needs a comment "
        "saying why it is safe this time"
    )
    assert "GITHUB_REF_NAME" in guard


def test_the_version_is_a_release_number():
    """Not a docstring quote, which is what the guard used to extract."""
    import re

    text = (ROOT / "src" / "zimablue" / "_version.py").read_text()
    match = re.search(r'^__version__ = "([^"]+)"$', text, re.M)
    assert match, "expected a single __version__ assignment on its own line"
    assert re.fullmatch(r"\d+\.\d+\.\d+([ab]\d+|rc\d+)?", match.group(1)), match.group(1)
    assert match.group(1) == zb.__version__


def test_every_publish_path_can_actually_publish():
    """A skipped job whose dependant is also skipped is a green tick and no upload.

    ``target: pypi`` used to skip the TestPyPI job, which skipped the verify
    job that needed it, which skipped the PyPI upload that needed *that* --
    so the option did nothing and reported success. Job conditions compose in
    a direction that is easy to get wrong and impossible to notice.
    """
    import yaml

    jobs = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text())["jobs"]

    def reachable(name: str, *, tag: bool, target: str) -> bool:
        """Would this job run, given how GitHub skips dependants of skips?"""
        condition = jobs[name].get("if")
        if condition is not None:
            fires = ("startsWith(github.ref, 'refs/tags/v')" in condition and tag) or (
                f"inputs.target == '{target}'" in condition
            )
            if not fires:
                return False
        needs = jobs[name].get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        return all(reachable(n, tag=tag, target=target) for n in needs)

    assert reachable("pypi", tag=True, target=""), "a tag should reach PyPI"
    assert reachable("pypi", tag=False, target="pypi"), "target=pypi should reach PyPI"
    assert not reachable("pypi", tag=False, target="testpypi"), "a dry run must not"
    assert reachable("testpypi", tag=False, target="testpypi")
    # Every path rehearses on TestPyPI first.
    for tag, target in ((True, ""), (False, "pypi"), (False, "testpypi")):
        assert reachable("testpypi", tag=tag, target=target)
