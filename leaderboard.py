import os
import json
import time
import datetime
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEADERBOARD_NAMESPACE = "main"
# Runtime records must not modify a tracked source file. Keeping them in an
# ignored, branch-specific local file lets Git switch branches after a run
# while keeping the main and codex score histories independent.
LEADERBOARD_NAMESPACE = os.environ.get(
    "KABOAT_LEADERBOARD_NAMESPACE", DEFAULT_LEADERBOARD_NAMESPACE)
LEADERBOARD_FILE = os.environ.get(
    "KABOAT_LEADERBOARD_FILE",
    os.path.join(PROJECT_DIR, ".kaboat_runtime",
                 f"leaderboard-{LEADERBOARD_NAMESPACE}.json"),
)
BENCHMARK_FILE = Path(PROJECT_DIR) / "leaderboard_benchmarks.json"

BENCHMARK_IDS = (f"{DEFAULT_LEADERBOARD_NAMESPACE}_avg",
                 f"{DEFAULT_LEADERBOARD_NAMESPACE}_best")


def load_benchmarks():
    """Read only this branch's two fixed results without rerunning simulations."""
    if not BENCHMARK_FILE.exists():
        return []
    with BENCHMARK_FILE.open(encoding="utf-8") as source:
        document = json.load(source)
    records = document["benchmarks"]
    if (document.get("branch") != DEFAULT_LEADERBOARD_NAMESPACE or
            len(records) != len(BENCHMARK_IDS) or
            {record.get("benchmark_id") for record in records} != set(BENCHMARK_IDS) or
            any(record.get("type") != "benchmark" for record in records)):
        raise ValueError("Benchmark file must contain this branch's two distinct benchmark records")
    return records

def load_leaderboard():
    """Load this branch's human records; old score files stay inactive."""
    if not os.path.exists(LEADERBOARD_FILE):
        return []
    try:
        with open(LEADERBOARD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        print(f"[Leaderboard] Load error: {e}")
        return []

def save_leaderboard(records):
    """Save human records to the branch-specific runtime file."""
    try:
        os.makedirs(os.path.dirname(LEADERBOARD_FILE), exist_ok=True)
        with open(LEADERBOARD_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[Leaderboard] Save error: {e}")
        return False

def add_record(collisions, arrival_time, cumulative_turn_deg=0.0, cum_turn=None, player_name="Player"):
    """
    신규 주행 완료 기록 추가 및 자동 정렬 저장
    1순위: 충돌 횟수 (적을수록 우수)
    2순위: 도달 시간 (빠를수록 우수)
    3순위: 누적 회전 각도 (적을수록 우수)
    """
    if cum_turn is not None:
        cumulative_turn_deg = cum_turn
    records = load_leaderboard()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    new_record = {
        "type": "player",
        "player": player_name,
        "collisions": int(collisions),
        "time": round(float(arrival_time), 2),
        "cumulative_turn_deg": round(float(cumulative_turn_deg), 1),
        "date": now_str,
        "timestamp": time.time()
    }
    
    records.append(new_record)
    save_leaderboard(records)
    return new_record

def get_unified_records():
    """Rank human and benchmark records by collisions, time, then turn."""
    records = load_leaderboard()
    all_entries = [dict(r) for r in records]
    all_entries.extend(dict(record) for record in load_benchmarks())
    all_entries.sort(key=lambda r: (
        r.get("collisions", 999),
        r.get("time", 9999.0),
        r.get("cumulative_turn_deg", 99999.0)
    ))
    return all_entries

def get_sorted_records():
    """1순위: 충돌, 2순위: 시간, 3순위: 누적회전각 기준으로 정렬된 전체 기록 반환"""
    return get_unified_records()

def get_top_records(limit=10):
    """상위 N개 통합 기록 반환"""
    return get_unified_records()[:limit]

def get_benchmark_rank(benchmark_id):
    """Full-list rank, even when the benchmark falls below the Top 10."""
    for rank, record in enumerate(get_unified_records(), 1):
        if record.get("type") == "benchmark" and record.get("benchmark_id") == benchmark_id:
            return rank
    return None


def get_display_records(limit=10):
    """Top 10 plus out-of-range benchmarks, each shown only once."""
    ranked = list(enumerate(get_unified_records(), 1))
    return ranked[:limit] + [item for item in ranked[limit:]
                             if item[1].get("type") == "benchmark"]

def get_player_rank(current_record):
    """통합 랭킹에서 현재 플레이어 기록의 순위(1-indexed) 계산"""
    if not current_record:
        return None
    unified = get_unified_records()
    cur_key = (current_record.get("collisions", 999), current_record.get("time", 9999.0), current_record.get("cumulative_turn_deg", 99999.0))
    cur_ts = current_record.get("timestamp", 0)
    for idx, r in enumerate(unified):
        if r.get("type") == "benchmark":
            continue
        r_key = (r.get("collisions", 999), r.get("time", 9999.0), r.get("cumulative_turn_deg", 99999.0))
        r_ts = r.get("timestamp", 0)
        if r_key == cur_key and abs(r_ts - cur_ts) < 0.05:
            return idx + 1
    # 만약 동일 타임스탬프를 못 찾을 경우 값 기반 순위 산출
    rank = 1
    for r in unified:
        r_key = (r.get("collisions", 999), r.get("time", 9999.0), r.get("cumulative_turn_deg", 99999.0))
        if r_key < cur_key:
            rank += 1
    return rank
