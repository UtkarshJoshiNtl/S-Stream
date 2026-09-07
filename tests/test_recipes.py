"""The recipe table, the recipe dialog and the preset library are one contract.

C16 found that the flagship vortex-shedding recipe resolved to nothing: the
action table keys on `karman_street` while the preset's display name is
`High-Re Wake`. Nothing tested any of it, so the breakage shipped.

The contract has three parts, all checked here without constructing a window:

  * `MainWindow._RECIPE_ACTIONS` and `recipes_dialog.RECIPES` are two separate
    hardcoded dicts describing the same five recipes. They must agree on keys,
    or the dialog offers a recipe with no action.
  * Every non-null action points at a preset id that exists in the library.
  * A recipe description that names a preset must name the one it loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from presets.loader import find_preset, list_presets, load_preset, validate_presets
from workbench.app import MainWindow
from workbench.dialogs.recipes_dialog import RECIPES

_PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets" / "scenes"


def _tagged_presets() -> list[tuple[str, str]]:
    """(filename, gallery caption) for every preset that declares a recipe."""
    out = []
    for f in sorted(_PRESETS_DIR.glob("*.json")):
        product = json.loads(f.read_text()).get("product", {})
        tag = product.get("recipe", "")
        if tag:
            out.append((f.name, tag))
    return out


class TestRecipeActionsResolve:
    def test_library_has_no_problems(self):
        assert validate_presets() == []

    def test_every_preset_has_an_id(self):
        for preset in list_presets():
            assert preset["id"], f"{preset['file']} has no id"

    def test_preset_ids_are_unique(self):
        ids = [p["id"] for p in list_presets()]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("label", sorted(MainWindow._RECIPE_ACTIONS))
    def test_recipe_preset_id_exists(self, label):
        action = MainWindow._RECIPE_ACTIONS[label]
        if action is None:
            pytest.skip(f"{label!r} is handled by a dialog, not a preset")
        preset_id = action[0]
        assert (
            find_preset(preset_id) is not None
        ), f"recipe {label!r} points at unknown preset id {preset_id!r}"

    def test_the_flagship_recipe_resolves_to_the_high_re_wake(self):
        """The exact case C16 fixed, pinned by name so the id cannot drift."""
        preset_id, _ = MainWindow._RECIPE_ACTIONS["Show vortex shedding"]
        preset = find_preset(preset_id)
        assert preset is not None
        assert preset["name"] == "High-Re Wake"

    def test_resolution_is_not_a_display_name_lookup(self):
        """Resolution keys on the id, so a display name must not resolve."""
        assert find_preset("High-Re Wake") is None


class TestDialogAndActionTablesAgree:
    """Two hardcoded dicts describing the same recipes."""

    def test_key_sets_match(self):
        assert set(RECIPES) == set(
            MainWindow._RECIPE_ACTIONS
        ), "the recipe dialog and the action table list different recipes"

    @pytest.mark.parametrize("label", sorted(RECIPES))
    def test_every_recipe_has_a_description(self, label):
        assert RECIPES[label].strip()

    def test_the_dialog_is_what_opens_the_recipes_tab(self):
        """`_open_recipes_dialog` shows StartDialog, so make sure the recipe
        list the user sees is the one under test."""
        assert "Show vortex shedding" in RECIPES


class TestDescriptionsMatchBehaviour:
    def test_named_preset_matches_the_loaded_preset(self):
        names = {p["name"]: p["id"] for p in list_presets()}
        for label, action in MainWindow._RECIPE_ACTIONS.items():
            description = RECIPES.get(label, "")
            mentioned = sorted(n for n in names if n in description)
            if not mentioned:
                continue
            loaded = find_preset(action[0])["name"] if action else None
            assert mentioned == [loaded], (
                f"recipe {label!r} tells the user to open {mentioned} "
                f"but loads {loaded!r}"
            )


class TestPresetMetadata:
    def test_some_presets_have_a_gallery_caption(self):
        assert len(_tagged_presets()) >= 8

    @pytest.mark.parametrize(
        "filename,caption", _tagged_presets(), ids=lambda v: str(v)[:28]
    )
    def test_caption_is_printable(self, filename, caption):
        assert caption.strip()
        assert caption == caption.strip()

    def test_every_preset_id_matches_its_filename(self):
        """Ids were seeded from the filenames; a divergence should be a
        deliberate decision rather than an accident."""
        for f in sorted(_PRESETS_DIR.glob("*.json")):
            product = json.loads(f.read_text()).get("product", {})
            assert (
                product.get("preset_id") == f.stem
            ), f"{f.name} declares id {product.get('preset_id')!r}"

    def test_load_preset_round_trips_the_id(self):
        for preset in list_presets():
            scene = load_preset(preset["file"])
            assert scene.product.preset_id == preset["id"]

    def test_display_names_are_not_identical_to_ids(self):
        """If they were, the old substring bug would have been invisible and
        could easily come back."""
        for preset in list_presets():
            assert preset["name"].lower() != preset["id"].lower()
