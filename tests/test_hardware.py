import sys
import types

import pytest

from src import hardware


class TestSelectWhisperModel:
    @pytest.mark.parametrize("gb,expected", [
        (32, "large-v3"),
        (16, "large-v3"),
        (15.9, "medium"),
        (8, "medium"),
        (7.9, "small"),
        (4, "small"),
        (3.9, "base"),
        (2, "base"),
        (1.9, "tiny"),
        (0, "tiny"),
    ])
    def test_thresholds(self, gb, expected):
        assert hardware.select_whisper_model(gb) == expected


class TestAutoSelectModel:
    def _mock_psutil(self, monkeypatch, available_bytes):
        fake_psutil = types.ModuleType("psutil")
        fake_psutil.virtual_memory = lambda: types.SimpleNamespace(available=available_bytes)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    def test_picks_model_from_available_ram(self, monkeypatch):
        self._mock_psutil(monkeypatch, 16 * 1024 ** 3)
        assert hardware.auto_select_model(installed=[]) == "large-v3"

    def test_low_ram_picks_tiny(self, monkeypatch):
        self._mock_psutil(monkeypatch, 1 * 1024 ** 3)
        assert hardware.auto_select_model(installed=[]) == "tiny"

    def test_divides_ram_across_parallel_workers(self, monkeypatch):
        # 16GB available / 4 workers = 4GB effective -> "small" tier
        self._mock_psutil(monkeypatch, 16 * 1024 ** 3)
        assert hardware.auto_select_model(parallel_workers=4, installed=[]) == "small"

    def test_zero_workers_does_not_divide_by_zero(self, monkeypatch):
        self._mock_psutil(monkeypatch, 16 * 1024 ** 3)
        assert hardware.auto_select_model(parallel_workers=0, installed=[]) == "large-v3"

    def test_missing_psutil_falls_back_to_small(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)
        assert hardware.auto_select_model(installed=[]) == "small"


class TestPreferInstalled:
    def test_downgrades_to_the_largest_model_already_on_disk(self):
        # 'auto' should just work: silently starting a 3 GB download (which
        # fails outright on a restricted network) is the opposite of that.
        assert hardware._prefer_installed("large-v3", ["small", "medium"]) == "medium"

    def test_keeps_the_hardware_pick_when_it_is_installed(self):
        assert hardware._prefer_installed("small", ["small", "medium"]) == "small"

    def test_never_exceeds_the_hardware_budget(self):
        # Only 'base' fits, and nothing that small is installed, so download it
        # rather than running a model the machine cannot hold.
        assert hardware._prefer_installed("base", ["small", "medium"]) == "base"

    def test_nothing_installed_returns_the_hardware_pick(self):
        assert hardware._prefer_installed("medium", []) == "medium"


class TestInstalledWhisperModels:
    def _make_model(self, cache_dir, name, with_weights=True):
        snapshot = cache_dir / f"models--Systran--faster-whisper-{name}" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        if with_weights:
            (snapshot / "model.bin").write_bytes(b"x")

    def test_detects_downloaded_models_smallest_first(self, tmp_path):
        self._make_model(tmp_path, "medium")
        self._make_model(tmp_path, "small")
        assert hardware.installed_whisper_models(tmp_path) == ["small", "medium"]

    def test_empty_cache(self, tmp_path):
        assert hardware.installed_whisper_models(tmp_path) == []

    def test_an_interrupted_download_does_not_count_as_installed(self, tmp_path):
        # The folder exists but the weights never landed.
        self._make_model(tmp_path, "large-v3", with_weights=False)
        assert hardware.installed_whisper_models(tmp_path) == []

    def test_missing_cache_directory(self, tmp_path):
        assert hardware.installed_whisper_models(tmp_path / "nope") == []


class TestAutoSelectPrefersInstalled:
    def test_auto_avoids_an_unnecessary_download(self, monkeypatch):
        fake_psutil = types.ModuleType("psutil")
        fake_psutil.virtual_memory = lambda: types.SimpleNamespace(available=32 * 1024 ** 3)
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        # 32 GB would allow large-v3, but only small is on disk.
        assert hardware.auto_select_model(installed=["small"]) == "small"
