"""Lossless episode PNG capture outside the simulation/render process."""
import atexit
import multiprocessing as mp
import pygame

_context = mp.get_context('spawn')
_queue = None
_worker = None


def _save_worker(jobs):
    from PIL import Image
    while True:
        item = jobs.get()
        if item is None:
            break
        size, pixels, path = item
        try:
            Image.frombytes('RGBA', size, pixels).save(path, format='PNG')
        except Exception as exc:
            print(f'[Warning] Episode screenshot failed: {exc}', flush=True)


def start_capture_worker():
    """Initialize before the GUI clock; process startup cannot stall a frame."""
    global _queue, _worker
    if _worker is None:
        _queue = _context.Queue()
        _worker = _context.Process(target=_save_worker,args=(_queue,),daemon=False)
        _worker.start()


def save_episode_frame(screen, path):
    start_capture_worker()
    # The exact finished RGBA frame becomes immutable before the surface is
    # reused. Encoding and disk I/O never run in the GUI process.
    _queue.put((screen.get_size(),pygame.image.tostring(screen,'RGBA'),str(path)))


def shutdown_capture_worker():
    global _queue, _worker
    if _worker is not None:
        _queue.put(None)
        _worker.join(timeout=30)
        _queue.close()
        _queue, _worker = None, None


atexit.register(shutdown_capture_worker)
