"""Branch-local fixed benchmark ranking and display rules."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import leaderboard


class LeaderboardTest(unittest.TestCase):
    def setUp(self):
        self.branch = leaderboard.DEFAULT_LEADERBOARD_NAMESPACE
        self.other = "main" if self.branch == "codex" else "codex"
        self.records = [
            dict(type="benchmark", benchmark_id=identifier,
                 player=identifier.upper(), collisions=0,
                 time=float(20 + index), cumulative_turn_deg=0)
            for index, identifier in enumerate(leaderboard.BENCHMARK_IDS)
        ]

    def write_benchmarks(self, path, branch=None, records=None):
        path.write_text(json.dumps({
            "branch": branch or self.branch,
            "benchmarks": self.records if records is None else records,
        }), encoding="utf-8")

    def test_only_local_benchmarks_keep_full_rank_below_top_ten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            players = root / "players.json"
            benchmarks = root / "benchmarks.json"
            players.write_text(json.dumps([
                dict(type="player", player=f"Player {i}", collisions=0,
                     time=float(i), cumulative_turn_deg=0, timestamp=i)
                for i in range(1, 14)
            ]), encoding="utf-8")
            self.write_benchmarks(benchmarks)
            with patch.object(leaderboard, "LEADERBOARD_FILE", str(players)), \
                    patch.object(leaderboard, "BENCHMARK_FILE", benchmarks):
                displayed = leaderboard.get_display_records()
                benchmark_rows = [(rank, row["benchmark_id"]) for rank, row in displayed
                                  if row.get("type") == "benchmark"]
                self.assertEqual(len(displayed), 12)
                self.assertEqual(benchmark_rows, [
                    (14, self.records[0]["benchmark_id"]),
                    (15, self.records[1]["benchmark_id"]),
                ])
                self.assertIsNone(leaderboard.get_benchmark_rank(f"{self.other}_avg"))
                leaderboard.add_record(0, 0.5, player_name="New player")
                self.assertEqual(leaderboard.get_benchmark_rank(
                    self.records[0]["benchmark_id"]), 15)
                self.assertEqual(json.loads(benchmarks.read_text())["benchmarks"],
                                 self.records)

    def test_benchmarks_inside_top_ten_are_not_duplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            benchmarks = root / "benchmarks.json"
            near_top = [dict(record, time=float(i + 1))
                        for i, record in enumerate(self.records)]
            self.write_benchmarks(benchmarks, records=near_top)
            with patch.object(leaderboard, "LEADERBOARD_FILE", str(root / "missing.json")), \
                    patch.object(leaderboard, "BENCHMARK_FILE", benchmarks):
                self.assertEqual([rank for rank, _ in leaderboard.get_display_records()], [1, 2])

    def test_opposite_branch_benchmarks_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            benchmarks = Path(folder) / "benchmarks.json"
            self.write_benchmarks(benchmarks, branch=self.other)
            with patch.object(leaderboard, "BENCHMARK_FILE", benchmarks):
                with self.assertRaises(ValueError):
                    leaderboard.load_benchmarks()
            other_records = [dict(record, benchmark_id=f"{self.other}_{i}")
                             for i, record in zip(("avg", "best"), self.records)]
            self.write_benchmarks(benchmarks, records=other_records)
            with patch.object(leaderboard, "BENCHMARK_FILE", benchmarks):
                with self.assertRaises(ValueError):
                    leaderboard.load_benchmarks()


if __name__ == "__main__":
    unittest.main()
