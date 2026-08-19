"""Plugin discovery in the registries."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

import zimablue.registry as registry_module
from zimablue.planners import PLANNERS
from zimablue.registry import Registry


def fake_entry(name, factory=None, error=None, dist="zbplug"):
    def load():
        if error is not None:
            raise error
        return factory

    return SimpleNamespace(name=name, load=load, dist=SimpleNamespace(name=dist))


def test_a_plugin_name_appears_without_being_imported(monkeypatch):
    loaded = []

    def factory():
        loaded.append(True)
        return "built"

    monkeypatch.setattr(
        registry_module, "_entry_points", lambda group: [fake_entry("extra", factory)]
    )
    registry = Registry("widget", entry_point_group="zimablue.widgets")
    registry.add("builtin", lambda: "builtin")

    assert registry.names() == ["builtin", "extra"]
    assert "extra" in registry
    assert not loaded, "listing names must not import the plugin"
    assert registry.create("extra") == "built"
    assert loaded


def test_a_plugin_cannot_shadow_a_builtin(monkeypatch):
    monkeypatch.setattr(
        registry_module, "_entry_points", lambda group: [fake_entry("builtin", lambda: "impostor")]
    )
    registry = Registry("widget", entry_point_group="zimablue.widgets")
    registry.add("builtin", lambda: "original")

    with pytest.warns(UserWarning, match="already taken"):
        names = registry.names()
    assert names == ["builtin"]
    assert registry.create("builtin") == "original"


def test_the_first_of_two_plugins_claiming_a_name_wins(monkeypatch):
    entries = [fake_entry("same", lambda: "first"), fake_entry("same", lambda: "second")]
    monkeypatch.setattr(registry_module, "_entry_points", lambda group: entries)
    registry = Registry("widget", entry_point_group="zimablue.widgets")

    with pytest.warns(UserWarning, match="already taken"):
        assert registry.create("same") == "first"


def test_a_broken_plugin_fails_at_create_and_names_its_package(monkeypatch):
    entry = fake_entry("broken", error=ModuleNotFoundError("no such module"), dist="zb-broken")
    monkeypatch.setattr(registry_module, "_entry_points", lambda group: [entry])
    registry = Registry("widget", entry_point_group="zimablue.widgets")

    assert "broken" in registry.names()
    with pytest.raises(ImportError, match="zb-broken"):
        registry.create("broken")


def test_a_plugin_that_is_not_a_factory_is_refused(monkeypatch):
    monkeypatch.setattr(
        registry_module, "_entry_points", lambda group: [fake_entry("odd", factory=42)]
    )
    registry = Registry("widget", entry_point_group="zimablue.widgets")

    with pytest.raises(TypeError, match="must name a callable"):
        registry.create("odd")


def test_discovery_reads_real_package_metadata(tmp_path, monkeypatch):
    """The mechanism end to end: a dist-info on sys.path, no mocking."""
    (tmp_path / "zbplugin_mod.py").write_text("def make():\n    return 'from the plugin'\n")
    info = tmp_path / "zbplugin-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: zbplugin\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text("[zimablue.test_widgets]\ncustom = zbplugin_mod:make\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    registry = Registry("widget", entry_point_group="zimablue.test_widgets")
    assert "custom" in registry.names()
    assert registry.create("custom") == "from the plugin"


def test_builtin_registries_are_unchanged_with_no_plugins_installed():
    assert "boustrophedon" in PLANNERS.names()
    assert PLANNERS.entry_point_group == "zimablue.planners"
