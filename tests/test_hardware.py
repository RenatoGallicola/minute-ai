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
        assert hardware.auto_select_model() == "large-v3"

    def test_low_ram_picks_tiny(self, monkeypatch):
        self._mock_psutil(monkeypatch, 1 * 1024 ** 3)
        assert hardware.auto_select_model() == "tiny"

    def test_divides_ram_across_parallel_workers(self, monkeypatch):
        # 16GB available / 4 workers = 4GB effective -> "small" tier
        self._mock_psutil(monkeypatch, 16 * 1024 ** 3)
        assert hardware.auto_select_model(parallel_workers=4) == "small"

    def test_zero_workers_does_not_divide_by_zero(self, monkeypatch):
        self._mock_psutil(monkeypatch, 16 * 1024 ** 3)
        assert hardware.auto_select_model(parallel_workers=0) == "large-v3"

    def test_missing_psutil_falls_back_to_small(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)
        assert hardware.auto_select_model() == "small"
