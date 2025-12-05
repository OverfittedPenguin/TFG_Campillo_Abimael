import casadi as ca
import numpy as np


class NLP_CRUISE:
    def __init__(self):
        # DEFINE STATE AND CONTROLS
        # 7 states [u,w,q,theta,x,z,m].
        # 2 controls [dt,de].
        # 1 end time [tF]
        self.x = ca.MX.sym("x", 7)   
        self.u = ca.MX.sym("u", 2) 
        self.utf = ca.MX.sym("utF", 1)

    @staticmethod
    def STATES_CONTROL_VECTOR(x, u, utf, N):
        # Retrieving number of states and controls vectors.
        # Empty states an controls vector for NLP.
        nx = x.size1()
        nu = u.size1()
        nutf = utf.size1()
        w = ca.MX()

        # Stack symbolic vectors for N nodes.
        for k in range(N):
            xk = ca.MX.sym(f"x{k}", nx)
            uk = ca.MX.sym(f"u{k}", nu)
            w = ca.vertcat(w, xk, uk)
        utf = ca.MX.sym(f"u_tF", nutf)
        w = ca.vertcat(w,utf)

        return w
    
    @staticmethod
    def COMPUTE_TRIM(ac,at,sim):
        # Decision variables for trimming.
        y = ca.SX.sym('y', 3) 
        alpha_sym = y[0]
        dt_sym    = y[1]
        de_sym    = y[2]

        # Known physical parameters. Target height, velocity and mass.
        V = sim.Vtp  
        h = -sim.w0[5] 
        m = sim.w0[6]
        
        # Calm wind atmosphere conditions.
        rho = at.ISA_RHO(h) 
        q_bar = 0.5 * rho * V**2

        # AERODYNAMIC COEFFICIENTS
        CL = ac.CL_0 + ac.CL_alpha * alpha_sym + ac.CL_de * de_sym
        CD = ac.CD_0 + ac.K * CL**2
        Cm = ac.Cm_0 + ac.Cm_alpha * alpha_sym + ac.Cm_de * de_sym

        # AERODYNAMIC FORCES AND MOMENT
        L = q_bar * ac.S * CL
        D = q_bar * ac.S * CD
        M_aero = q_bar * ac.S * ac.c * Cm

        # PROPULSIVE FORCES AND MOMENT
        T_max, M_T_max = ac.PROPULSIVE_FORCES_MOMENTS(V, ac.RPM, rho, alpha_sym)
        T = dt_sym * T_max
        M_prop = dt_sym * M_T_max

        # EQUATIONS (simplified using small angles)
        Fx = T * ca.cos(alpha_sym) - D 
        Fz = L + T * ca.sin(alpha_sym) - m * at.g 
        My = M_aero + M_prop

        # State boundaries.
        lbx = [ac.lb[3], ac.lb[7], ac.lb[8]]
        ubx = [ac.ub[3], ac.ub[7], ac.ub[8]] 

        # NLP PROBLEM DEFINITION
        f = ca.sumsqr(ca.vertcat(Fx, Fz, My))
        opts = {}
        opts['ipopt.max_iter'] = 1000
        opts['ipopt.tol'] = 1e-12
        opts['ipopt.acceptable_tol'] = 1e-12
        nlp = {'x': y, 'f': f}
        solver = ca.nlpsol('trim_solver_nlp', 'ipopt', nlp, opts)

        # INITIAL GUESS
        y0 = np.array([np.deg2rad(2.0), 0.5, 0.0]) 

        # SOLVER
        sol = solver(
            x0=y0,   
            lbx=lbx, 
            ubx=ubx  
        )
        
        # RESULTS AND DECOMPOSITION INTO w0
        y_opt = sol['x'].full().flatten()
        alpha_trim = y_opt[0]
        dt_trim    = y_opt[1]
        de_trim    = y_opt[2]
        
        u = V * np.cos(alpha_trim)
        w = V * np.sin(alpha_trim)
        
        w0_state = np.array([u, w, 0.0, alpha_trim, 0.0, -h, m])
        w0_control = [0.5, 0.05, sim.t0]
        w0 = ca.vertcat(w0_state, w0_control)
        
        return w0
        
    @staticmethod
    def DYNAMIC_EQUATIONS(w,ac,at,sim):
        # States and control retrieving.
        x1 = w[0]
        x2 = w[1]
        x3 = w[2]
        x4 = w[3]
        x5 = w[4]
        x6 = w[5]
        x7 = w[6]
        u1 = w[7]
        u2 = w[8]

        # Computation of atmosphere parameters,
        # aerodynamic velocity and dynamic pressure.
        rho = at.ISA_RHO(-x6)
        ua = x1 - sim.wind[0]
        wa = x2 - sim.wind[1]
        alpha = ca.atan2(wa, ua)
        V = ca.sqrt(ua**2 + wa**2)
        q_bar = 0.5 * rho * V**2

        # AERODYNAMIC COEFFICIENTS
        CL = ac.CL_0 + ac.CL_alpha * alpha + ac.CL_de * u2
        CD = ac.CD_0 + ac.K * CL**2
        Cm = ac.Cm_0 + ac.Cm_alpha * alpha + ac.Cm_de * u2

        # PROPULSIVE FORCE AND MOMENT
        T_max,M_T_max = ac.PROPULSIVE_FORCES_MOMENTS(V,ac.RPM,rho,alpha)
        T_max = ca.if_else(x7 > (ac.BEM + ac.PM),T_max,0.0)
        M_T_max = ca.if_else(x7 > (ac.BEM + ac.PM),M_T_max,0.0)
        T = u1 * T_max
        M_T = u1 * M_T_max
        Fb_prop = ca.vertcat(T, 0.0)

        # Rotation matrices from body axes to wind axes and to local horizon axes.
        Rsb = ca.vertcat(
            ca.horzcat(ca.cos(alpha), ca.sin(alpha)),
            ca.horzcat(-ca.sin(alpha), ca.cos(alpha))
        )
    
        Rhb = ca.vertcat(
            ca.horzcat(ca.cos(x4), ca.sin(x4)),
            ca.horzcat(-ca.sin(x4), ca.cos(x4))
        )

        # FORCES
        Cb_aero = ca.mtimes(Rsb.T,ca.vertcat(-CD, -CL))
        Fb_aero = ac.S * q_bar * Cb_aero
        Fb_grav = ca.mtimes(Rhb.T,ca.vertcat(0.0, x7 * at.g))
        Fb_total = Fb_aero + Fb_prop + Fb_grav

        # MOMENTS
        Mb_aero = q_bar * ac.S * ac.c * Cm
        Mb_prop = M_T
        Mb_total = Mb_aero + Mb_prop
        Iyy = ac.Iyy

        # EQUATIONS OF MOTION
        uk = Fb_total[0] / x7 - x2 * x3
        wk = Fb_total[1] / x7 + x1 * x3
        qk = Mb_total / Iyy
        thetak = x3
        mk = ca.if_else(x7 > (ac.BEM + ac.PM),-ac.SFC,0.0)
        
        # KINEMATICS. Inertial velocities computation.
        inertial_vel = ca.mtimes(Rhb, ca.vertcat(x1, x2))
        xk = inertial_vel[0]
        zk = inertial_vel[1]

        return [uk,wk,qk,thetak,xk,zk,mk]
    
    @staticmethod
    def DYNAMIC_CONSTRAINTS(wi,wj,fi,fj,dT,tF):
        g_dyn = []
        lbg_dyn = []
        ubg_dyn = []
        for k in range(wi.size1()):
            # Dynamic constraints. Equality constraints.
            g_dyn.append(wj[k] - wi[k] - dT/2 * tF * (fi[k] + fj[k]))
            lbg_dyn.append(0)
            ubg_dyn.append(0)

        return g_dyn, lbg_dyn, ubg_dyn
    
    @staticmethod
    def PATH_CONSTRAINTS(w,sim,N):
        g_path = []
        lbg_path = []
        ubg_path = []

        for k in range(N-1):
            wi = w[9*k:9*(k+1)]
            # TARGET POINT VELOCITY CONSTRAINT
            ua = wi[0] - sim.wind[0]
            wa = wi[1] - sim.wind[1]

            # Path constraints. Inequality constraints.
            g_path.append(0.9*sim.Vtp - ca.sqrt(ua**2 + wa**2))
            lbg_path.append(-1e20)
            ubg_path.append(0)
            g_path.append(ca.sqrt(ua**2 + wa**2) - 1.1*sim.Vtp)
            lbg_path.append(-1e20)
            ubg_path.append(0)   

            # TARGET ALTITUDE CONSTRAINT
            href_l = -sim.w0[5] - 2.5
            href_u = -sim.w0[5] + 2.5
            hi = -wi[5]

            # Path constraints. Inequality constraints.
            g_path.append(href_l - hi)
            lbg_path.append(-1e20)
            ubg_path.append(0)
            g_path.append(hi - href_u)
            lbg_path.append(-1e20)
            ubg_path.append(0)   

        # END TIME CONSTRAINT
        x0 = w[4]
        xf = w[9*N-5]

        g_path.append(sim.Vtp*sim.t_lb - xf + x0)
        lbg_path.append(-1e20)
        ubg_path.append(0)
        g_path.append(xf - x0 - sim.Vtp*sim.t_ub)
        lbg_path.append(-1e20)
        ubg_path.append(0)   

        return g_path, lbg_path, ubg_path
    
    @staticmethod
    def INITIAL_AND_FINAL_CONSTRAINTS(w,w0,wf,N):
        # INITIAL STATE CONSTRAINTS. Controls freed.
        w_0 = w[:7]
        g_0 = []
        lbg_0 = []
        ubg_0 = []

        # Initial state. Equality constraints.
        for k in range(w_0.size1()):
            g_0.append(w_0[k] - w0[k])
            lbg_0.append(0)
            ubg_0.append(0)

        # FINAL STATE CONSTRAINTS. Controls freed.
        w_f = w[7*(N-1):7*N]
        g_f = []
        lbg_f = []
        ubg_f = []
        if wf != 0:
            # Final state. Equality constraints.
            for k in range(w_f.size1()):
                g_f.append(w_f[k] - wf[k])
                lbg_f.append(0)
                ubg_f.append(0)

        return g_0, g_f, lbg_0, lbg_f, ubg_0, ubg_f

    @staticmethod
    def SIMPLE_BOUNDS(lb,ub,sim,N):
        lbx = []
        ubx = []

        # Iteration to asign simple bounds to each state and control variable.
        for k in range(N):
            for j in range(9):
                lbx.append(lb[j])
                ubx.append(ub[j])
        lbx.append(sim.t_lb)
        ubx.append(sim.t_ub)

        return lbx, ubx     

    @staticmethod
    def CONSTRAINTS_AND_BOUNDS(x,u,utf,ac,at,sim):
        # Time step, number of nodes and state bounds.
        dT = sim.dT
        N = sim.N
        lb = ac.lb
        ub = ac.ub

        # STATE AND CONTROL VECTORS
        w = NLP_CRUISE.STATES_CONTROL_VECTOR(x,u,utf,N)
        tF = w[9*N]

        # Computation of trim condition.
        w0_trim_node = NLP_CRUISE.COMPUTE_TRIM(ac,at,sim)
        sim.w0 = w0_trim_node[:9]
        
        # Reconstruction of full initial guess
        # as planar functions.
        w0_ls = []
        for k in range(N):
            w0_ls.append(w0_trim_node[0])
            w0_ls.append(w0_trim_node[1])
            w0_ls.append(w0_trim_node[2])
            w0_ls.append(w0_trim_node[3])
            w0_ls.append(w0_trim_node[0] * np.cos(w0_trim_node[3]) * dT * k)
            w0_ls.append(w0_trim_node[5])
            w0_ls.append(w0_trim_node[6])
            w0_ls.append(w0_trim_node[7])
            w0_ls.append(w0_trim_node[8])
        w0_ls.append(ca.DM(sim.t0))
        w0 = np.concatenate(w0_ls)

        # DYNAMIC COSNTRAINTS HANDLING
        g_dyn = []
        lbg_dyn = []
        ubg_dyn = []
        for k in range(N-1):
            wi = w[9*k:9*(k+1)]
            wj = w[9*(k+1):9*(k+2)]
            fi = NLP_CRUISE.DYNAMIC_EQUATIONS(wi,ac,at,sim)
            fj = NLP_CRUISE.DYNAMIC_EQUATIONS(wj,ac,at,sim)               
            g, lbg, ubg = NLP_CRUISE.DYNAMIC_CONSTRAINTS(wi[:7],wj[:7],fi[:7],fj[:7],dT,tF)
            g_dyn.append(g)
            lbg_dyn.append(lbg)
            ubg_dyn.append(ubg)
        g_dyn = sum(g_dyn,[])

        # PATH CONSTRAINTS HANDLING
        g_path, lbg_path, ubg_path = NLP_CRUISE.PATH_CONSTRAINTS(w,sim,N)

        # INITIAL AND FINAL STATE CONSTRAINTS HANDLING
        g_0, g_f, lbg_0, lbg_f, ubg_0, ubg_f = NLP_CRUISE.INITIAL_AND_FINAL_CONSTRAINTS(w,sim.w0,sim.wf,N)

        # SIMPLE BOUNDS FOR STATES AND CONTROL
        lbx, ubx = NLP_CRUISE.SIMPLE_BOUNDS(lb,ub,sim,N)

        # Construction of final vectors.
        g = g_dyn + g_path + g_0 + g_f
        lbg = lbg_dyn + lbg_path + lbg_0 + lbg_f
        ubg = ubg_dyn + ubg_path + ubg_0 + ubg_f

        return w0, w, lbx, ubx, g, lbg, ubg
    
    @staticmethod
    def COST_FUCNTIONAL(w,ac,at,sim):
        # Cost initialisation, time step and number
        # of nodes.
        J = 0
        dT = sim.dT
        N = sim.N
        tF = w[9*N]
        for k in range(N-1):
            # Weights assignation for gamma, gamma dot and controls.
            wg = sim.cruise_wg[0]
            wh = sim.cruise_wg[1]
            wg_dot = sim.cruise_wg[2]
            wdt = sim.cruise_wg[3]
            wde = sim.cruise_wg[4]

            # Normalisation vars.
            g_max = np.deg2rad(12.0)
            g_dot_max = g_max / sim.dT
            de_max = np.deg2rad(12.0)
            href = -sim.w0[5]

            # Actual and next states and functions.
            wi = w[9*k:9*(k+1)]
            wj = w[9*(k+1):9*(k+2)]
            fi = NLP_CRUISE.DYNAMIC_EQUATIONS(wi,ac,at,sim)
            fj = NLP_CRUISE.DYNAMIC_EQUATIONS(wj,ac,at,sim)

            # Alpha computation and theta retrieving.
            ai = ca.atan2(wi[1],wi[0])
            thi = wi[3]
            aj = ca.atan2(wj[1],wj[0])
            thj = wj[3]
            
            # Gamma computation.
            gi = thi - ai
            gj = thj - aj

            # Gamma dot computation.
            gi_dot = wi[2] - (fi[1]*wi[0] - fi[0]*wi[1]) / (wi[0]**2 + wi[1]**2)
            gj_dot = wj[2] - (fj[1]*wj[0] - fj[0]*wj[1]) / (wj[0]**2 + wj[1]**2)

            # Altitude computation.
            hi = -wi[5]
            hj = -wj[5]

            # COST FUCNTIONAL (Minimisation of gamma, gamma dot and controls)
            J += dT/2 * (wg*(gi**2 + gj**2) / g_max**2 + wg_dot*(gi_dot**2 + gj_dot**2) / g_dot_max**2  + wh*((hi - href)**2 / href**2 + (hj - href)**2 / href**2)) + wde*(wj[8] - wi[8])**2 / (sim.dT*de_max**2) + wdt*(wj[7] - wi[7])**2 / sim.dT
        J += tF
        return J
