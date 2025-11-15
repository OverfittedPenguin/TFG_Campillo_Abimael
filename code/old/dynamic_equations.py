import numpy as np
from aircraft import Aircraft
from atmosphere import Atmos

class DynEqs:
    def __init__(
        self,
        aircraft,
        atmos,
        control=None
    ):
        self.aircraft = aircraft
        self.atmos = atmos
        self.control = control if control is not None else {"elevator": 0.0, "throttle": 0.0}

    def EOM_3DOF(self, t, state, PHASE: str):
        # RETRIEVING PARAMETERS
        ac = self.aircraft
        at = self.atmos
        ctrls = self.control

        # Unpack state.
        u, w, q, theta, x, z = state

        # Relative atmosphere velocity.
        wsp = at.Wind
        ua, wa = u - wsp[0], w - wsp[1]
        Va = np.sqrt(ua**2 + wa**2)
        alpha = np.arctan2(wa, ua)
        V = np.sqrt(u**2 + w**2)

        # ATMOSPHERIC PROPERTIES
        rho,sigma = at.ISA_RHO(z)
        q_bar = 0.5 * rho * Va**2

        # AERODYNAMIC COEFFICIENTS
        CL = ac.CL_0 + ac.CL_alpha * alpha + ac.CL_de * ctrls["elevator"]
        CD = ac.CD_0 + ac.K * CL**2
        Cm = ac.Cm_0 + ac.Cm_alpha * alpha + ac.Cm_de * ctrls["elevator"]

        # PROPULSIVE FORCE AND MOMENT
        T_max,M_T,P = ac.PROPULSIVE_FORCES_MOMENTS(V,n_RPM,rho,sigma,alpha)
        T = ctrls["throttle"] * T_max
        Fb_prop = np.array([T, 0.0])

        # Rotation matrices from body axes to local horizon axes.
        Rsb = np.array([
            [np.cos(alpha), np.sin(alpha)],
            [-np.sin(alpha), np.cos(alpha)]
        ])
        Rhb = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

        # Mass computation and inertia assignment.
        mass = ac.BEM + ac.FM * (1 - ac.SFC * P * t) + ac.PM
        Iyy = ac.Iyy

        # FORCES
        Cb_aero = Rsb.T @ np.array([-CD, -CL])
        Fb_aero = q_bar * ac.S * Cb_aero
        Fb_grav = Rhb.T @ np.array([0.0, mass * at.g])
        Fb_total = Fb_aero + Fb_prop + Fb_grav

        # MOMENTS
        Mb_aero = q_bar * ac.S * ac.c * Cm
        Mb_prop = M_T
        Mb_total = Mb_aero + Mb_prop

        # EQUATIONS OF MOTION
        u_dot = Fb_total[0] / mass - w * q
        w_dot = Fb_total[1] / mass + u * q
        q_dot = Mb_total / Iyy
        theta_dot = q

        # Kinematics. Inertial velocities computation.
        inertial_vel = Rhb @ np.array([u, w])
        x_dot, z_dot = inertial_vel

        return np.array([u_dot, w_dot, q_dot, theta_dot, x_dot, z_dot])
        