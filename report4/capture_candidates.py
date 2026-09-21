import os
import sys
import pygame

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from environment import BoatEnv
import main

OUT_DIR = '/home/soonhong/kaboat/report4/candidate_screenshots'
os.makedirs(OUT_DIR, exist_ok=True)

orig_render = BoatEnv.render
captured = []

def hooked_render(self, hits):
    orig_render(self, hits)
    
    frame = getattr(self, 'frame', 0)
    has_bezier = getattr(self, 'bezier_path', None) is not None and len(self.bezier_path) > 3
    cand_wps = getattr(self, 'candidate_wps', [])
    in_action = 120 <= frame <= 650
    
    # Capture when the boat is actively curving through buoys with rich telemetry
    if in_action and has_bezier and len(cand_wps) >= 1 and (frame % 70 == 0):
        c_idx = len(captured) + 1
        
        # 1. 2D Cockpit view
        shot_2d = f"candidate_{c_idx}_2d_cockpit_frame{frame}.png"
        path_2d = os.path.join(OUT_DIR, shot_2d)
        pygame.image.save(self.screen, path_2d)
        captured.append({
            "id": f"2D-{c_idx}",
            "file": shot_2d,
            "path": path_2d,
            "frame": frame,
            "type": "2D 콕핏 인터페이스",
            "desc": "2D 수역 주행 맵 + 3차 베지어 회피 곡선 + 하단 실시간 텔레메트리 콕핏 계측 패널"
        })
        print(f"Captured: {shot_2d}")

        # 2. 3D Chase camera view
        try:
            self.fullscreen_3d = True
            self.cam_3d_mode = 1 # 3rd person chase
            orig_render(self, hits)
            shot_3d_chase = f"candidate_{c_idx}_3d_chase_frame{frame}.png"
            path_3d_chase = os.path.join(OUT_DIR, shot_3d_chase)
            pygame.image.save(self.screen, path_3d_chase)
            captured.append({
                "id": f"3D-Chase-{c_idx}",
                "file": shot_3d_chase,
                "path": path_3d_chase,
                "frame": frame,
                "type": "3D 체이스 뷰 (3인칭 추종)",
                "desc": "선체 후방 3인칭 추종 시점: 파도와 부표 사이를 고속 통과하는 역동적 3D 뷰"
            })
            print(f"Captured: {shot_3d_chase}")
        except Exception as e:
            print("3D chase capture error:", e)

        # 3. 3D Tactical Drone view
        try:
            self.cam_3d_mode = 2 # Tactical drone
            orig_render(self, hits)
            shot_3d_drone = f"candidate_{c_idx}_3d_drone_frame{frame}.png"
            path_3d_drone = os.path.join(OUT_DIR, shot_3d_drone)
            pygame.image.save(self.screen, path_3d_drone)
            captured.append({
                "id": f"3D-Drone-{c_idx}",
                "file": shot_3d_drone,
                "path": path_3d_drone,
                "frame": frame,
                "type": "3D 전술 드론 쿼터뷰",
                "desc": "상공 쿼터뷰 시점: 수역 전체 부표 배치와 선체 항로를 한눈에 조망하는 3D 뷰"
            })
            print(f"Captured: {shot_3d_drone}")
        except Exception as e:
            print("3D drone capture error:", e)

        # Restore 2D
        self.fullscreen_3d = False
        orig_render(self, hits)

    if len(captured) >= 9:
        # Exit after capturing enough diverse candidates
        raise StopIteration("Captured enough candidates!")

BoatEnv.render = hooked_render

if __name__ == '__main__':
    try:
        main.run()
    except (StopIteration, SystemExit):
        pass
    finally:
        pygame.quit()
        print(f"\nSuccessfully finished! Total screenshots captured: {len(captured)}")
        for c in captured:
            print(f"[{c['id']}] {c['file']} ({c['type']})")
