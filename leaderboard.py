import os
import json
import time
import datetime

LEADERBOARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.json")

# 자율운항 GAP 알고리즘 및 라인트레이싱 알고리즘 벤치마크 기준치
DEFAULT_AI_BENCHMARKS = [
    {
        "name": "GAP 알고리즘",
        "player": "GAP 알고리즘",
        "collisions": 0,
        "time": 48.9,
        "speed": 34.6,
        "cumulative_turn_deg": 415.4,
        "is_ai": True,
        "date": "GAP 평균"
    },
    {
        "name": "라인트레이싱",
        "player": "라인트레이싱",
        "collisions": 0,
        "time": 66.8,
        "speed": 31.4,
        "cumulative_turn_deg": 1039.0,
        "is_ai": True,
        "date": "라인트레이싱 평균"
    }
]

def get_ai_benchmarks():
    """Report 5 또는 Report 2 벤치마크 통계 파일에서 최신 알고리즘 평균 데이터 동적 로드"""
    r5_summary = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report5", "benchmark_5000_summary.json")
    r2_summary = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report2", "benchmark_5000_summary.json")
    
    target_summary = r5_summary if os.path.exists(r5_summary) else (r2_summary if os.path.exists(r2_summary) else None)
    
    if target_summary and os.path.exists(target_summary):
        try:
            with open(target_summary, "r", encoding="utf-8") as f:
                data = json.load(f)
                gn = data.get("gapnav", {})
                lt = data.get("linetrace", {})
                return [
                    {
                        "name": "GAP 알고리즘",
                        "player": "GAP 알고리즘",
                        "collisions": int(round(gn.get("collision_count", 0) / max(1, gn.get("total_episodes", 1)))),
                        "time": round(float(gn.get("time_sec_mean", 48.9)), 2),
                        "speed": round(float(gn.get("mean_speed_mean", 34.6)), 1),
                        "cumulative_turn_deg": round(float(gn.get("cum_turn_deg_mean", 415.4)), 1),
                        "is_ai": True,
                        "date": "GAP 평균"
                    },
                    {
                        "name": "라인트레이싱",
                        "player": "라인트레이싱",
                        "collisions": int(round(lt.get("collision_count", 0) / max(1, lt.get("total_episodes", 1)))),
                        "time": round(float(lt.get("time_sec_mean", 66.8)), 2),
                        "speed": round(float(lt.get("mean_speed_mean", 31.4)), 1),
                        "cumulative_turn_deg": round(float(lt.get("cum_turn_deg_mean", 1039.0)), 1),
                        "is_ai": True,
                        "date": "라인트레이싱 평균"
                    }
                ]
        except Exception:
            pass
    return DEFAULT_AI_BENCHMARKS

# 이전 호환성 유지용 객체
AI_BENCHMARK = DEFAULT_AI_BENCHMARKS[0]

def load_leaderboard():
    """leaderboard.json 파일에서 주행 기록 목록 로드"""
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
    """주행 기록 목록을 leaderboard.json 파일에 저장"""
    try:
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
    """사용자 주행 기록과 자율운항 AI 벤치마크(GAP 알고리즘, 라인트레이싱)를 통합하여 1, 2, 3순위로 정렬된 전체 기록 반환"""
    records = load_leaderboard()
    all_entries = [dict(r) for r in records]
    ai_list = get_ai_benchmarks()
    for ai_entry in ai_list:
        all_entries.append(dict(ai_entry))
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

def get_ai_benchmark_rank(name="GAP 알고리즘"):
    """통합 랭킹에서 지정된 AI 알고리즘의 순위(1-indexed) 계산"""
    unified = get_unified_records()
    for idx, r in enumerate(unified):
        if r.get("is_ai", False) and (r.get("name") == name or r.get("player") == name):
            return idx + 1
    return 1

def get_player_rank(current_record):
    """통합 랭킹에서 현재 플레이어 기록의 순위(1-indexed) 계산"""
    if not current_record:
        return None
    unified = get_unified_records()
    cur_key = (current_record.get("collisions", 999), current_record.get("time", 9999.0), current_record.get("cumulative_turn_deg", 99999.0))
    cur_ts = current_record.get("timestamp", 0)
    for idx, r in enumerate(unified):
        if r.get("is_ai", False):
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
