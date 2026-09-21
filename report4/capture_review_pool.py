import os
import sys
import math
import pygame

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from environment import BoatEnv
import main

OUT_DIR_2D = '/home/soonhong/kaboat/report4/review_pool/2d_cockpit'
OUT_DIR_3D = '/home/soonhong/kaboat/report4/review_pool/3d_wide_cropped'
os.makedirs(OUT_DIR_2D, exist_ok=True)
os.makedirs(OUT_DIR_3D, exist_ok=True)

orig_render = BoatEnv.render
captured_2d = []
captured_3d = []

# Frame milestones for capturing varied positions along the course
# e.g., early approach, entering first buoys, middle obstacle maze, narrow gap, exit, near goal
TARGET_FRAMES = [60, 110, 160, 210, 260, 320, 390, 480, 580]
captured_milestones = set()

def hooked_render(self, hits):
    orig_render(self, hits)
    
    frame = getattr(self, 'frame', 0)
    
    # Find nearest target frame not yet captured
    for tf in TARGET_FRAMES:
        if abs(frame - tf) <= 2 and tf not in captured_milestones:
            captured_milestones.add(tf)
            idx = len(captured_milestones)
            boat_x = int(self.boat_pos[0])
            boat_y = int(self.boat_pos[1])
            spd_kt = round(math.hypot(self.boat_vel[0], self.boat_vel[1]) * 1.94384, 1)
            
            # -------------------------------------------------------------
            # GROUP 1: Default 2D Full Cockpit View (1840 x 920)
            # -------------------------------------------------------------
            f_2d_name = f"2D_pos{idx}_frame{frame}_x{boat_x}.png"
            p_2d = os.path.join(OUT_DIR_2D, f_2d_name)
            pygame.image.save(self.screen, p_2d)
            captured_2d.append({
                "idx": idx,
                "file": f_2d_name,
                "path": p_2d,
                "frame": frame,
                "boat_x": boat_x,
                "boat_y": boat_y,
                "spd": spd_kt,
                "state": getattr(self, 'state_str', 'NAV')
            })
            print(f"[2D Captured #{idx}] {f_2d_name} (X={boat_x}, Spd={spd_kt}kt)")

            # -------------------------------------------------------------
            # GROUP 2: Wide 3D Panoramic View (Cropped 1840 x 644, no bottom panel)
            # -------------------------------------------------------------
            # 2-A: 3D Chase Camera (3인칭 추종)
            try:
                self.fullscreen_3d = True
                self.cam_3d_mode = 1 # Chase 3rd
                orig_render(self, hits)
                
                # Crop just the 3D top area (0, 0, w, sim_h)
                sub_3d_chase = self.screen.subsurface((0, 0, self.w, self.sim_h)).copy()
                f_3d_chase = f"3D_Chase_pos{idx}_frame{frame}_x{boat_x}.png"
                p_3d_chase = os.path.join(OUT_DIR_3D, f_3d_chase)
                pygame.image.save(sub_3d_chase, p_3d_chase)
                captured_3d.append({
                    "idx": idx,
                    "mode": "Chase (3인칭 추종)",
                    "file": f_3d_chase,
                    "path": p_3d_chase,
                    "frame": frame,
                    "boat_x": boat_x
                })
                print(f"[3D Chase Captured #{idx}] {f_3d_chase}")
            except Exception as e:
                print(f"3D chase error: {e}")

            # 2-B: 3D Tactical Drone Camera (상공 전술 쿼터뷰)
            try:
                self.cam_3d_mode = 2 # Drone tactical
                orig_render(self, hits)
                
                sub_3d_drone = self.screen.subsurface((0, 0, self.w, self.sim_h)).copy()
                f_3d_drone = f"3D_Drone_pos{idx}_frame{frame}_x{boat_x}.png"
                p_3d_drone = os.path.join(OUT_DIR_3D, f_3d_drone)
                pygame.image.save(sub_3d_drone, p_3d_drone)
                captured_3d.append({
                    "idx": idx,
                    "mode": "Drone (전술 쿼터뷰)",
                    "file": f_3d_drone,
                    "path": p_3d_drone,
                    "frame": frame,
                    "boat_x": boat_x
                })
                print(f"[3D Drone Captured #{idx}] {f_3d_drone}")
            except Exception as e:
                print(f"3D drone error: {e}")

            # Restore to 2D
            self.fullscreen_3d = False
            orig_render(self, hits)
            break

    if len(captured_milestones) >= len(TARGET_FRAMES):
        raise StopIteration("All milestones captured!")

BoatEnv.render = hooked_render

if __name__ == '__main__':
    try:
        main.run()
    except (StopIteration, SystemExit):
        pass
    finally:
        pygame.quit()
        print("\nAll screenshot captures completed!")
        print(f"2D Cockpit Views: {len(captured_2d)} files in {OUT_DIR_2D}")
        print(f"3D Wide Cropped Views: {len(captured_3d)} files in {OUT_DIR_3D}")
