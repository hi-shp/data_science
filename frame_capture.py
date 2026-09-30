"""Save a copied episode frame without blocking the simulation loop."""
import atexit
import multiprocessing as mp

import pygame

_queue = None
_worker = None


def _save_worker(jobs):
    while True:
        item = jobs.get()
        if item is None:
            break
        size, pixels, path = item
        try:
            pygame.image.save(pygame.image.frombuffer(pixels, size, 'RGBA'), path)
        except Exception as exc:
            print(f'[Warning] Episode screenshot failed: {exc}', flush=True)


def start_capture_worker():
    global _queue, _worker
    if _worker is None:
        context = mp.get_context('spawn')
        _queue = context.Queue()
        _worker = context.Process(target=_save_worker, args=(_queue,), daemon=False)
        _worker.start()


def save_episode_frame(screen, path):
    start_capture_worker()
    _queue.put((screen.get_size(), pygame.image.tostring(screen, 'RGBA'), str(path)))


def shutdown_capture_worker():
    global _queue, _worker
    if _worker is not None:
        _queue.put(None)
        _worker.join(timeout=30)
        _queue.close()
        _queue = _worker = None


atexit.register(shutdown_capture_worker)
