"""MAIN-identical buoy clustering, accelerated only for display annotations."""
import numpy as np
from numba import njit
from config import GRID


@njit(cache=True)
def component_labels(gx, gy):
    """Same row-ordered union/find as MAIN's extract_clusters_from_grid."""
    n = len(gx)
    parent = np.arange(n)
    for r in range(n):
        for c in range(r+1, n):
            dx, dy = gx[r]-gx[c], gy[r]-gy[c]
            if dx*dx+dy*dy > 17:
                continue
            pr, pc = r, c
            while parent[pr] != pr:
                pr = parent[pr]
            while parent[pc] != pc:
                pc = parent[pc]
            if pr != pc:
                parent[pr] = pc
    labels = np.empty(n, dtype=np.int32)
    root_label = np.full(n, -1, dtype=np.int32)
    count = 0
    for i in range(n):
        root = i
        while parent[root] != root:
            root = parent[root]
        if root_label[root] == -1:
            root_label[root] = count
            count += 1
        labels[i] = root_label[root]
    return labels, count


class MainDisplayClusters:
    def __init__(self):
        self.key = None
        self.labels = None
        self.count = 0

    def extract(self, grid):
        gy, gx = np.where(grid >= 1.)
        n = len(gx)
        if n == 0:
            return []
        if n == 1:
            return [np.array([gx[0]*GRID+GRID*.5, gy[0]*GRID+GRID*.5], dtype=np.float32)]
        key = (gx.tobytes(), gy.tobytes())
        if key != self.key:
            self.key = key
            self.labels, self.count = component_labels(gx, gy)
        weights = grid[gy, gx]
        sum_w = np.bincount(self.labels, weights=weights, minlength=self.count)
        valid = sum_w > 0
        inv_w = 1./sum_w[valid]
        cx = (np.bincount(self.labels, weights=weights*gx, minlength=self.count)[valid]*inv_w)*GRID+GRID*.5
        cy = (np.bincount(self.labels, weights=weights*gy, minlength=self.count)[valid]*inv_w)*GRID+GRID*.5
        return list(np.column_stack((cx,cy)).astype(np.float32))


def warmup():
    component_labels(np.zeros(1,dtype=np.int64),np.zeros(1,dtype=np.int64))
