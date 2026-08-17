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


def test_the_animated_logo_is_swapped_for_a_still():
    """PyPI only serves description images through its camo proxy, whose SVG
    handling is not worth betting the hero image on."""
    from hatch_build import absolutise

    text = absolutise((ROOT / "README.md").read_text())
    assert "logo-animated.svg" not in text
    assert "docs/assets/logo.png" in text
    assert not [u for u in re.findall(r'<img[^>]+src="([^"]+)"', text) if u.endswith(".svg")]


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
    from hatch_build import absolutise

    html = renderer.render(absolutise((ROOT / "README.md").read_text()))
    assert html is not None, "PyPI would reject this description"
    assert html.count("<img") >= 5, "images were stripped by the sanitiser"
    assert "<table" in html and "<pre" in html
