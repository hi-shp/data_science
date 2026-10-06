import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('KABOAT_WIDTH','1800')
import unittest
from types import SimpleNamespace
import numpy as np
from navigation_map import NavigationMap
from control_path import smooth_path,lookahead
from perception import lidar_hits_np
from route_planner import route_target
from boat_control import ControllerParameters
from vessel_dynamics import VesselParameters


class NavigationTests(unittest.TestCase):
    def sensor(self, obstacles, position=(3.,6.)):
        angles=np.linspace(-np.pi,np.pi,180,endpoint=False,dtype=np.float32)
        pos=np.array(position)
        ranges,*_=lidar_hits_np(pos*50,0,angles,np.array(obstacles).reshape(-1,3)*50,320,(0,0,1800,630))
        m=NavigationMap(36,12.6);m.observe(pos,0,angles,ranges/50,6.4,.04)
        return m,ranges

    def env(self,m,position=(3.,6.)):
        return SimpleNamespace(navigation_map=m,dynamics=VesselParameters(),control=ControllerParameters(),boat_pos=np.array(position)*50,map_w=1800,sim_h=630,target=np.array([1700.,300.]),frame=1,lidar_range=320,current_wp=None)

    def test_unseen_obstacle_does_not_change_route(self):
        a=[[7,5,.35]];b=a+[[12,6,.4]]
        m1,r1=self.sensor(a);m2,r2=self.sensor(b)
        np.testing.assert_array_equal(r1,r2)
        e1,e2=self.env(m1),self.env(m2)
        np.testing.assert_array_equal(route_target(e1),route_target(e2))
        np.testing.assert_array_equal(e1.raw_route,e2.raw_route)
        np.testing.assert_array_equal(e1.control_path,e2.control_path)
        m3,_=self.sensor(a,position=(7,6));m4,_=self.sensor(b,position=(7,6))
        e3,e4=self.env(m3,(7,6)),self.env(m4,(7,6))
        route_target(e3);route_target(e4)
        self.assertFalse(np.array_equal(e3.raw_route,e4.raw_route))

    def test_controller_is_independent_of_hidden_world_objects(self):
        from boat_control import select_command
        results=[]
        for world in ([[7,5,.35]], [[7,5,.35],[12,6,.4]]):
            m,r=self.sensor(world);e=self.env(m)
            e.dt=.04;e.boat_heading=0.;e.command_yaw_rate=0.
            e.lidar_dists=r;e.perceived_obstacles=m.obstacles*50
            e.physics_state=lambda:np.array([3.,6.,0.,0.,0.,0.,0.,0.])
            # No dynamic_obstacles/ground-truth attribute is exposed to control.
            command=select_command(e)
            results.append((command,e.command_speed,e.heading_target))
        np.testing.assert_array_equal(results[0],results[1])

    def test_missing_curve_keeps_hull_checked_recovery_available(self):
        from unittest.mock import patch
        from boat_control import select_command
        m,r=self.sensor([]);e=self.env(m)
        e.dt=.04;e.boat_heading=0.;e.command_yaw_rate=0.
        e.lidar_dists=r;e.perceived_obstacles=m.obstacles*50
        e.physics_state=lambda:np.array([3.,6.,0.,0.,0.,0.,0.,0.])
        with patch('boat_control.route_target',return_value=None):
            select_command(e)
        self.assertLess(e.command_speed,0.)
        self.assertFalse(e.emergency_mode)
        self.assertGreater(e.predicted_clearance,e.control.safety_margin_m)

    def test_occluded_obstacle_does_not_leak(self):
        m1,r1=self.sensor([[6,6,.7]])
        m2,r2=self.sensor([[6,6,.7],[8,6,.25]])
        np.testing.assert_array_equal(r1,r2)
        np.testing.assert_allclose(m1.obstacles,m2.obstacles)

    def test_memory_expires_and_unknown_is_not_free(self):
        m,_=self.sensor([[7,5,.35]])
        self.assertFalse(m.known_free[20,-1])
        angles=np.linspace(-np.pi,np.pi,180,endpoint=False)
        m.observe(np.array([30.,6.]),0,angles,np.full(180,6.4),6.4,6.)
        self.assertFalse(np.any(m.obstacles[:,0]<10))
        self.assertFalse(m.known_free[20,10])

    def test_curve_preserves_clearance_and_rounds_corner(self):
        raw=np.array([[1.,1.],[3.,1.],[3.,3.]])
        safe=lambda p:bool(np.all(np.linalg.norm(p-np.array([2.,2.]),axis=1)>.75))
        p=smooth_path(raw,safe)
        self.assertIsNotNone(p);self.assertTrue(safe(p))
        v=np.diff(p,axis=0);v/=np.linalg.norm(v,axis=1)[:,None]
        angle=np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1,1))
        self.assertLess(np.max(angle),.15)

    def test_continuous_lookahead_at_vertex(self):
        p=np.array([[0.,0.],[2.,0.],[2.,2.]])
        a,_=lookahead(p,np.array([.9999,0]),1.)
        b,_=lookahead(p,np.array([1.0001,0]),1.)
        self.assertLess(np.linalg.norm(a-b),.0003)

    def test_gap_changes_corridor_without_changing_physics(self):
        m,_=self.sensor([]);a,b=self.env(m),self.env(m)
        b.current_wp={'pos':np.array([350.,200.]),'pair':(1,2)}
        route_target(a);route_target(b)
        self.assertIsNotNone(b.selected_gap)
        self.assertFalse(np.array_equal(a.raw_route,b.raw_route))
        self.assertFalse(np.array_equal(a.control_path,b.control_path))
        self.assertGreater(np.linalg.norm(a.pursuit_target-b.pursuit_target),1.)

    def test_reset_drops_all_perception_and_path_state(self):
        from environment import BoatEnv
        e=BoatEnv(headless=True);e.navigation_map=object();e.control_path=np.ones((2,2));e.reset()
        self.assertFalse(hasattr(e,'navigation_map'));self.assertFalse(hasattr(e,'control_path'))
        e.close()

if __name__=='__main__':unittest.main()
