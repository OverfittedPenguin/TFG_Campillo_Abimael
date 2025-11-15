import casadi as ca
import numpy as np

class NLP_CRUISE:
    def __init__(self):
        # DEFINE STATE AND CONTROLS
        # 7 states [u,w,q,theta,x,z,m].
        # 2 controls [dt,de].
        self.x = ca.SX.sym("x", 7)   
        self.u = ca.SX.sym("u", 2) 

    @staticmethod
    def STATES_CONTROL_VECTOR(x, u, N):
        # Retrieving number of states and controls vectors.
        # Empty states an controls vector for NLP.
        nx = x.size1()
        nu = u.size1()
        w = ca.SX()

        # Stack symbolic vectors for N intervals.
        for k in range(N):
            xk = ca.SX.sym(f"x_{k}", nx)
            uk = ca.SX.sym(f"u_{k}", nu)
            w = ca.vertcat(w, xk, uk)
        return w
    
    @staticmethod
    def DYNAMIC_EQUATIONS(w,ac,at):
        # States and control retrieving.
        x0 = w[0]
        x1 = w[1]
        x2 = w[2]
        x3 = w[3]
        x4 = w[4]
        x5 = w[5]
        x6 = w[6]
        u0 = w[7]
        u1 = w[8]

        if ac.FLAPS == 0.0:
            ac.CL_0 = ac.CL_CD_F0[0]
            ac.CL_alpha = ac.CL_CD_F0[1]
            ac.CD_0 = ac.CL_CD_F0[2]
        elif ac.FLAPS == 15.0:
            ac.CL_0 = ac.CL_CD_F15[0]
            ac.CL_alpha = ac.CL_CD_F15[1]
            ac.CD_0 = ac.CL_CD_F15[2]
        elif ac.FLAPS == 40.0:
            ac.CL_0 = ac.CL_CD_F40[0]
            ac.CL_alpha = ac.CL_CD_F40[1]
            ac.CD_0 = ac.CL_CD_F40[2]
        else:
            raise ValueError(f"Flaps set doesn't met expected values F0, F15 or F40.")

        # Computation of atmosphere parameters,
        # aerodynamic velocity and dynamic pressure.
        rho = at.ISA_RHO(x5)
        ua, wa = x0-at.Wind[0], x1-at.Wind[1]
        alpha = np.arctan2(wa, ua)
        V = np.sqrt(ua**2 + wa**2)
        q_bar = 0.5*rho*V**2

        # AERODYNAMIC COEFFICIENTS
        CL = ac.CL_0 + ac.CL_alpha * alpha + ac.CL_de * u1
        CD = ac.CD_0 + ac.K * CL**2
        Cm = ac.Cm_0 + ac.Cm_alpha * alpha + ac.Cm_de * u1

        # PROPULSIVE FORCE AND MOMENT
        T_max,M_T = ac.PROPULSIVE_FORCES_MOMENTS(V,ac.RPM,rho,alpha)
        T = u0 * T_max
        Fb_prop = np.array([T, 0.0])

        # Rotation matrices from body axes to local horizon axes.
        Rsb = np.array([
            [np.cos(alpha), np.sin(alpha)],
            [-np.sin(alpha), np.cos(alpha)]
        ])
        Rhb = np.array([
            [np.cos(x3), np.sin(x3)],
            [-np.sin(x3), np.cos(x3)]
        ])

        # FORCES
        Cb_aero = Rsb.T @ np.array([-CD, -CL])
        Fb_aero = q_bar * ac.S * Cb_aero
        Fb_grav = Rhb.T @ np.array([0.0, x6 * at.g])
        Fb_total = Fb_aero + Fb_prop + Fb_grav

        # MOMENTS
        Mb_aero = q_bar * ac.S * ac.c * Cm
        Mb_prop = M_T
        Mb_total = Mb_aero + Mb_prop
        Iyy = ac.Iyy

        # EQUATIONS OF MOTION
        uk = Fb_total[0] / x6 - x1 * x2
        wk = Fb_total[1] / x6 + x0 * x2
        qk = Mb_total / Iyy
        thetak = x2
        mk = -ac.SFC

        # Kinematics. Inertial velocities computation.
        inertial_vel = Rhb @ np.array([x0, x1])
        xk, zk = inertial_vel

        return [uk,wk,qk,thetak,xk,zk,mk]
    
    @staticmethod
    def DYNAMIC_CONSTRAINTS(wi,wj,fi,fj,dT):
        g_dyn = []
        lbg_dyn = []
        ubg_dyn = []
        for k in range(wi.size1()):
            # Dynamic constraints. Equality constraints.
            g_dyn.append(wj[k] - wi[k] - dT/2 * (fi[k]+fj[k]))
            lbg_dyn.append(0)
            ubg_dyn.append(0)

        return g_dyn, lbg_dyn, ubg_dyn
    
    @staticmethod
    def PATH_CONSTRAINTS(wi,at,sim):
        # TARGET POINT VELOCITY CONSTRAINT
        ua, wa = wi[0]-at.Wind[0], wi[1]-at.Wind[1]
        g_path = []
        lbg_path = []
        ubg_path = []

        # Path constraints. Inqueality constraints.
        g_path.append(0.9*sim.Vtp - np.sqrt(ua**2 + wa**2))
        lbg_path.append(-1e20) # Negative infinity.
        ubg_path.append(0)
        g_path.append(np.sqrt(ua**2 + wa**2) - 1.1*sim.Vtp)
        lbg_path.append(-1e20) # Negative infinity.
        ubg_path.append(0)

        return g_path, lbg_path, ubg_path
