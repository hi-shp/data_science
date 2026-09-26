"""Compiled equivalent of the existing constant-command predictive horizon.

Unlike the three-step sampling prototype, production evaluates each fixed
speed/yaw command with 0.2 s physical prediction steps. Every hull and wall
check remains at the original prediction cadence.
"""
import math
import numpy as np
from experiments.fast_rollout import parameter_vector

try:
    from numba import njit
except ImportError:
    njit = None


if njit is not None:
    @njit(cache=True)
    def compiled_constant_rollout(initial, speeds, rates, obstacles, width, height,
                                  dt, horizon, safety_margin, params):
        count = len(speeds)
        z = np.empty((count,8),dtype=np.float64)
        trajectory = np.empty((count,horizon,8),dtype=np.float64)
        closest = np.full(count,10.)
        blocked = np.zeros(count,dtype=np.bool_)
        turn_cost = np.zeros(count)
        mass,inertia = params[0],params[1]
        surge_lin,surge_quad = params[2],params[3]
        sway_lin,sway_quad = params[4],params[5]
        yaw_lin,yaw_quad = params[6],params[7]
        arm,max_thrust = params[8],params[9]
        alpha = -math.expm1(-dt/params[10])
        speed_response,yaw_gain = params[11],params[12]
        for i in range(count):
            x,y,h,u,v,r,fl,fr = (initial[0],initial[1],initial[2],initial[3],
                                 initial[4],initial[5],initial[6],initial[7])
            for t in range(horizon):
                force = surge_lin*u+surge_quad*u*abs(u)+mass*(speeds[i]-u)/speed_response
                moment = yaw_lin*r+yaw_quad*r*abs(r)+yaw_gain*(rates[i]-r)
                diff = max(-max_thrust,min(max_thrust,moment/(2*arm)))
                common = max(-max_thrust+abs(diff),
                             min(max_thrust-abs(diff),force/2))
                left = max(-max_thrust,min(max_thrust,common-diff))
                right = max(-max_thrust,min(max_thrust,common+diff))
                next_fl = fl+alpha*(left-fl)
                next_fr = fr+alpha*(right-fr)
                c,s = math.cos(h),math.sin(h)
                dh1 = r
                du1 = (next_fl+next_fr-surge_lin*u-surge_quad*u*abs(u))/mass+v*r
                dv1 = (-sway_lin*v-sway_quad*v*abs(v))/mass-u*r
                dr1 = ((next_fr-next_fl)*arm-yaw_lin*r-yaw_quad*r*abs(r))/inertia
                hm,um,vm,rm = h+.5*dt*dh1,u+.5*dt*du1,v+.5*dt*dv1,r+.5*dt*dr1
                cm,sm = math.cos(hm),math.sin(hm)
                x += dt*(um*cm-vm*sm)
                y += dt*(um*sm+vm*cm)
                h += dt*rm
                u += dt*((next_fl+next_fr-surge_lin*um-surge_quad*um*abs(um))/mass+vm*rm)
                v += dt*((-sway_lin*vm-sway_quad*vm*abs(vm))/mass-um*rm)
                r += dt*(((next_fr-next_fl)*arm-yaw_lin*rm-yaw_quad*rm*abs(rm))/inertia)
                fl,fr = next_fl,next_fr
                trajectory[i,t,0],trajectory[i,t,1] = x,y
                trajectory[i,t,2],trajectory[i,t,3] = h,u
                trajectory[i,t,4],trajectory[i,t,5] = v,r
                trajectory[i,t,6],trajectory[i,t,7] = fl,fr
                if len(obstacles):
                    c,s = math.cos(h),math.sin(h)
                    margin = math.inf
                    for j in range(len(obstacles)):
                        dx,dy = obstacles[j,0]-x,obstacles[j,1]-y
                        along,lateral = dx*c+dy*s,-dx*s+dy*c
                        gap = max(-.56-along,along-.52,0.)
                        side = min(abs(lateral-.22),abs(lateral+.22))
                        margin = min(margin,math.hypot(gap,side)-obstacles[j,2]-.32)
                    closest[i] = min(closest[i],margin)
                    blocked[i] = blocked[i] or margin<safety_margin
                wall = min(y-.93,height-.93-y,x-.93,width-.93-x)
                closest[i] = min(closest[i],wall)
                blocked[i] = blocked[i] or wall<.05
                turn_cost[i] += r*r*dt
            for j in range(8):
                z[i,j] = trajectory[i,horizon-1,j]
        return z,closest,blocked,turn_cost,trajectory
else:
    compiled_constant_rollout = None


def warmup_constant_rollout():
    if compiled_constant_rollout is not None:
        from vessel_dynamics import VesselParameters
        compiled_constant_rollout(np.zeros(8),np.zeros(1),np.zeros(1),
                                  np.empty((0,3)),36.,12.6,.2,1,.2,
                                  parameter_vector(VesselParameters()))
