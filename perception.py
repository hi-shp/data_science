import math
import numpy as np
from config import GRID, GRID_W, GRID_H

# 고속 연산을 위한 그리드 인덱스 배열 및 라이다 각도 테이블 사전 생성
_Y_INDICES, _X_INDICES = np.indices((GRID_H, GRID_W), dtype=np.float32)
_Y_FLAT = _Y_INDICES.ravel()
_X_FLAT = _X_INDICES.ravel()

_REL_ANGLES = np.linspace(-np.pi, np.pi, 180, endpoint=False, dtype=np.float32)
_COS_REL = np.cos(_REL_ANGLES)[:, None]
_SIN_REL = np.sin(_REL_ANGLES)[:, None]

def lidar_hits_np(boat_pos, boat_heading, rel_angles, obstacles, lidar_range, map_bounds=None):
    n = len(rel_angles)
    ch = math.cos(boat_heading)
    sh = math.sin(boat_heading)
    vx = ch * _COS_REL - sh * _SIN_REL
    vy = sh * _COS_REL + ch * _SIN_REL

    x0, y0 = boat_pos

    if len(obstacles) > 0:
        ox = obstacles[:, 0]
        oy = obstacles[:, 1]
        orad = obstacles[:, 2]
        px = ox - x0
        py = oy - y0
        p_sq = px * px + py * py
        max_reach = lidar_range + orad
        cand = p_sq < (max_reach * max_reach)
        if np.any(cand):
            px_c = px[cand][None, :]
            py_c = py[cand][None, :]
            orad_c = orad[cand][None, :]
            p_sq_c = p_sq[cand][None, :]
            base_c = orad_c * orad_c - p_sq_c
            b = px_c * vx + py_c * vy
            disc = base_c + b * b
            mask = (b > 0) & (disc >= 0)
            t = np.where(mask, b - np.sqrt(np.maximum(0.0, disc)), lidar_range)
            d_final = np.min(t, axis=1).astype(np.float32)
        else:
            d_final = np.full(n, lidar_range, dtype=np.float32)
    else:
        d_final = np.full(n, lidar_range, dtype=np.float32)

    # 맵 외곽 벽(Boundary Walls)을 장애물로 인식 (목적지 방향 정면 수직벽 xmax 제외)
    if map_bounds is not None:
        xmin, ymin, xmax, ymax = map_bounds
        t_left = np.where(vx < -1e-5, (xmin - x0) / np.minimum(vx, -1e-5), lidar_range)
        # 목적지 방향의 정면 수직벽(xmax)은 벽으로 인식하지 않음
        t_top = np.where(vy < -1e-5, (ymin - y0) / np.minimum(vy, -1e-5), lidar_range)
        t_bottom = np.where(vy > 1e-5, (ymax - y0) / np.maximum(vy, 1e-5), lidar_range)

        t_left = np.where(t_left > 0, t_left, lidar_range)
        t_top = np.where(t_top > 0, t_top, lidar_range)
        t_bottom = np.where(t_bottom > 0, t_bottom, lidar_range)

        t_wall = np.minimum(t_left, np.minimum(t_top, t_bottom))
        d_final = np.minimum(d_final, t_wall[:, 0].astype(np.float32))
    
    # 벡터화된 히트 좌표 연산 (Python for 루프 제거)
    vx_flat = vx[:, 0]
    vy_flat = vy[:, 0]
    hits_x = x0 + vx_flat * d_final
    hits_y = y0 + vy_flat * d_final
    valid = d_final < lidar_range
    # 유효하지 않은 히트점은 NaN으로 마스킹 (렌더러에서 None 대신 NaN 검사)
    hits_x_out = np.where(valid, hits_x, np.nan).astype(np.float32)
    hits_y_out = np.where(valid, hits_y, np.nan).astype(np.float32)

    return d_final, hits_x_out, hits_y_out

def init_grid():
    return np.zeros((GRID_H, GRID_W), dtype=np.float32)

def update_grid(grid, hits_x, hits_y):
    # 벡터화 입력: hits_x, hits_y는 numpy float32 배열이며 유효하지 않은 인덱스는 NaN
    valid_mask = np.isfinite(hits_x)
    if not np.any(valid_mask):
        return
    hx = hits_x[valid_mask]
    hy = hits_y[valid_mask]
    gx = (hx * (1.0 / GRID)).astype(np.intp)
    gy = (hy * (1.0 / GRID)).astype(np.intp)
    bounds = (gx >= 0) & (gx < GRID_W) & (gy >= 0) & (gy < GRID_H)
    gx = gx[bounds]
    gy = gy[bounds]
    if len(gx) == 0:
        return
    # 국소 영역만 클램핑하여 전체 27,000 셀 순회 오버헤드 제거
    np.add.at(grid, (gy, gx), 1.0)
    grid[gy, gx] = np.minimum(grid[gy, gx], 20.0)

def extract_clusters_from_grid(grid):
    OCC = 1.0
    gy, gx = np.where(grid >= OCC)
    n = len(gx)
    if n == 0:
        return []
    if n == 1:
        return [np.array([gx[0] * GRID + GRID * 0.5, gy[0] * GRID + GRID * 0.5], dtype=np.float32)]
        
    weights = grid[gy, gx]
    # 그리드 정수 좌표계에서 (dx^2 + dy^2 <= 17) 인접 행렬 벡터화 연산 (cKDTree 및 파이썬 루프 100% 제거)
    dx = gx[:, None] - gx[None, :]
    dy = gy[:, None] - gy[None, :]
    adj = (dx * dx + dy * dy) <= 17
    # 초고속 분리 집합(Disjoint Set Union)을 통한 연결 요소 산출 (scipy 오버헤드 100% 제거)
    parent = list(range(n))
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i
    rows, cols = np.where(np.triu(adj, 1))
    if len(rows) > 0:
        rows_l = rows.tolist()
        cols_l = cols.tolist()
        for r, c in zip(rows_l, cols_l):
            pr = find(r)
            pc = find(c)
            if pr != pc:
                parent[pr] = pc
    label_map = {}
    labels = np.empty(n, dtype=np.int32)
    next_lbl = 0
    for i in range(n):
        root = find(i)
        lbl = label_map.get(root)
        if lbl is None:
            lbl = next_lbl
            label_map[root] = lbl
            next_lbl += 1
        labels[i] = lbl
    n_comp = next_lbl
    
    # bincount를 통한 가중 중심점 고속 벡터화 계산 후 월드 좌표 변환
    sum_w = np.bincount(labels, weights=weights, minlength=n_comp)
    valid = sum_w > 0
    inv_w = 1.0 / sum_w[valid]
    cx = (np.bincount(labels, weights=weights * gx, minlength=n_comp)[valid] * inv_w) * GRID + GRID * 0.5
    cy = (np.bincount(labels, weights=weights * gy, minlength=n_comp)[valid] * inv_w) * GRID + GRID * 0.5
    return list(np.column_stack((cx, cy)).astype(np.float32))

def match_clusters(prev_clusters, prev_ids, new_clusters, max_dist=28.0):
    if len(new_clusters) == 0:
        return [], []
    if len(prev_clusters) == 0 or len(prev_ids) == 0:
        return new_clusters, list(range(len(new_clusters)))
    
    n_new = len(new_clusters)
    n_prev = len(prev_clusters)
    
    # 거리 행렬을 벡터 연산으로 일괄 계산
    new_arr = np.array(new_clusters).reshape(n_new, 2)
    prev_arr = np.array(prev_clusters).reshape(n_prev, 2)
    diff = new_arr[:, None, :] - prev_arr[None, :, :]
    dist_mat = np.sum(diff * diff, axis=2)  # 제곱 거리로 비교하여 sqrt 연산 제거
    max_dist_sq = max_dist * max_dist
    
    new_ids = [0] * n_new
    maxid = max(prev_ids) + 1 if prev_ids else 0
    
    # 탐욕적 매칭: 전역 최소 거리 순서로 할당 (Python 내부 루프 최소화)
    flat_order = np.argsort(dist_mat, axis=None)
    assigned_new = np.zeros(n_new, dtype=bool)
    assigned_prev = np.zeros(n_prev, dtype=bool)
    
    for flat_idx in flat_order:
        i = int(flat_idx // n_prev)
        j = int(flat_idx % n_prev)
        if assigned_new[i] or assigned_prev[j]:
            continue
        if dist_mat[i, j] >= max_dist_sq:
            break  # 이후 모든 거리는 max_dist 이상이므로 중단
        new_ids[i] = prev_ids[j]
        assigned_new[i] = True
        assigned_prev[j] = True
    
    # 미매칭된 새 클러스터에 고유 ID 할당
    for i in range(n_new):
        if not assigned_new[i]:
            new_ids[i] = maxid
            maxid += 1
            
    return new_clusters, new_ids