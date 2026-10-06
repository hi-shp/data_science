"""Default startup must use the identical existing V2.1 opt-in path."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pygame
import main


class DefaultModeTests(unittest.TestCase):
    def boot(self, overrides):
        env = SimpleNamespace(clock=Mock(), sim_speed=1, manual_mode=False,
                              paused=False, show_leaderboard=False,
                              fullscreen_3d=False, obstacles=[])
        env.renderer = SimpleNamespace(engine_3d=None)
        with patch.dict(os.environ, overrides, clear=True), \
             patch.object(main, 'BoatEnv', return_value=env), \
             patch.object(main, 'start_capture_worker'), \
             patch('heavy.motion_v2.HeavyMotionV2') as core, \
             patch('heavy.motion_v2.warmup') as warmup, \
             patch.object(main, 'MomentumGapRouter') as legacy, \
             patch.object(pygame.event, 'get', return_value=[SimpleNamespace(type=pygame.QUIT)]), \
             patch.object(pygame, 'quit'):
            main.run()
            core.assert_called_once_with(env)
            warmup.assert_called_once_with(env)
            core.return_value.start.assert_called_once_with(env)
            core.return_value.close.assert_called_once_with()
            legacy.assert_not_called()
            return core.return_value.method_calls

    def test_default_and_explicit_opt_in_have_identical_startup(self):
        self.assertEqual([c[0] for c in self.boot({})],
                         [c[0] for c in self.boot({'MAIN_HEAVY_MOMENTUM_GAP': '1'})])


if __name__ == '__main__':
    unittest.main()
