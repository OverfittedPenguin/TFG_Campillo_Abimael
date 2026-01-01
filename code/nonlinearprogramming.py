import casadi as ca
import numpy as np

class NLP_STG1:
    # MISSION DESCENT STAGE
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
        V = sim.x0_USER[0]  
        h = -sim.lb[3] 
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
        opts['ipopt.max_iter'] = 3000
        opts['ipopt.tol'] = 1e-6
        opts['ipopt.acceptable_tol'] = 1e-6
        nlp = {'x': y, 'f': f}
        solver = ca.nlpsol('trim_solver_nlp', 'ipopt', nlp, opts)

        # INITIAL GUESS
        y0 = np.array([np.deg2rad(2.0), 0.5, 0.0]) 

        # SOLVER
        sol = solver(
            x0 = y0,   
            lbx = lbx, 
            ubx = ubx  
        )
        
        # RESULTS AND DECOMPOSITION INTO w0
        y_opt = sol['x'].full().flatten()
        alpha_trim = y_opt[0]
        dtps_trim    = y_opt[1]
        de_trim    = y_opt[2]
        
        u = V * np.cos(alpha_trim)
        w = V * np.sin(alpha_trim)
        
        w0_state = np.array([u, w, 0.0, alpha_trim, 0.0, -h, m])
        w0_control = [dtps_trim, de_trim, sim.t0[0]]
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
        mk = ca.if_else(x7 > (ac.BEM + ac.PM), -ac.SFC, 0.0) # SFC.
        
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
    def PATH_CONSTRAINTS(w,sim,ac,N):
        g_path = []
        lbg_path = []
        ubg_path = []

        for k in range(N-1):
            wi = w[9*k:9*(k+1)]
            # Aerodynamic velocity computation.
            ua = wi[0] - sim.wind[0]
            wa = wi[1] - sim.wind[1]
            V = ca.sqrt(ua**2 + wa**2)

            # VELOCITIES CONSTRAINTS
            VS = ac.lb_USER[0]
            VNO = ac.ub_USER[0]

            # Path constraints. Inequality constraints.
            g_path.append(VS - V)
            lbg_path.append(-1e20)
            ubg_path.append(0)
            g_path.append(V - VNO)
            lbg_path.append(-1e20)
            ubg_path.append(0)   

        # END TIME CONSTRAINT. Reference altitude and distance.
        href = -sim.ub[3]
        xref = sim.R_entry - sim.f_tp*sim.Vtp*sim.lb[1]
        hf = -w[9*N-4]
        xf = w[9*N-5]
        x0 = w[4]

        # Path constraints. Equality constraint.
        g_path.append(hf - href)
        lbg_path.append(0)
        ubg_path.append(0)  

        # Path constraint. Equality constraint.
        g_path.append((xf - x0) - xref)
        lbg_path.append(-1e20)
        ubg_path.append(0)   

        return g_path, lbg_path, ubg_path
    
    @staticmethod
    def INITIAL_STATE_CONSTRAINTS(w,w0):
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

        return g_0, lbg_0, ubg_0

    @staticmethod
    def SIMPLE_BOUNDS(lb,ub,sim,N):
        lbx = []
        ubx = []

        # Iteration to asign simple bounds to each state and control variable.
        for k in range(N):
            for j in range(9):
                lbx.append(lb[j])
                ubx.append(ub[j])
        lbx.append(sim.lb[0])
        ubx.append(sim.ub[0])

        return lbx, ubx     

    @staticmethod
    def CONSTRAINTS_AND_BOUNDS(x,u,utf,ac,at,sim):
        # Time step, number of nodes and state bounds.
        dT = sim.dT
        N = sim.N
        lb = ac.lb
        ub = ac.ub

        # STATE AND CONTROL VECTORS
        w = NLP_STG1.STATES_CONTROL_VECTOR(x,u,utf,N)
        tF = w[9*N]

        # Computation of trim condition.
        w0_trim = NLP_STG1.COMPUTE_TRIM(ac,at,sim)
        sim.w0 = w0_trim[:9]
        
        # Reconstruction of full initial guess
        # as planar functions.
        w0_ls = []
        for k in range(N):
            w0_ls.append(w0_trim[0])
            w0_ls.append(w0_trim[1])
            w0_ls.append(w0_trim[2])
            w0_ls.append(w0_trim[3])
            w0_ls.append(w0_trim[0] * np.cos(w0_trim[3]) * dT * k)
            w0_ls.append(w0_trim[5])
            w0_ls.append(w0_trim[6])
            w0_ls.append(w0_trim[7])
            w0_ls.append(w0_trim[8])
        w0_ls.append(ca.DM(sim.t0[0]))
        w0 = np.concatenate(w0_ls)

        # DYNAMIC COSNTRAINTS HANDLING
        g_dyn = []
        lbg_dyn = []
        ubg_dyn = []
        for k in range(N-1):
            wi = w[9*k:9*(k+1)]
            wj = w[9*(k+1):9*(k+2)]
            fi = NLP_STG1.DYNAMIC_EQUATIONS(wi,ac,at,sim)
            fj = NLP_STG1.DYNAMIC_EQUATIONS(wj,ac,at,sim)               
            g, lbg, ubg = NLP_STG1.DYNAMIC_CONSTRAINTS(wi[:7],wj[:7],fi[:7],fj[:7],dT,tF)
            g_dyn.append(g)
            lbg_dyn.append(lbg)
            ubg_dyn.append(ubg)
        g_dyn = sum(g_dyn,[])

        # PATH CONSTRAINTS HANDLING
        g_path, lbg_path, ubg_path = NLP_STG1.PATH_CONSTRAINTS(w,sim,ac,N)

        # INITIAL STATE CONSTRAINTS HANDLING
        g_0, lbg_0, ubg_0 = NLP_STG1.INITIAL_STATE_CONSTRAINTS(w,sim.w0)

        # SIMPLE BOUNDS FOR STATES AND CONTROL
        lbx, ubx = NLP_STG1.SIMPLE_BOUNDS(lb,ub,sim,N)

        # Construction of final vectors.
        g = g_dyn + g_path + g_0 
        lbg = lbg_dyn + lbg_path + lbg_0
        ubg = ubg_dyn + ubg_path + ubg_0

        return w0, w, lbx, ubx, g, lbg, ubg
    
    @staticmethod
    def COST_FUNCTIONAL(w,ac,at,sim):
        # Cost initialisation, time step and number
        # of nodes.
        J = 0
        dT = sim.dT
        N = sim.N
        tF = w[9*N]

        # Weights assignation.
        wg_tF = sim.STG1_wg[0]
        wg_g = sim.STG1_wg[1]
        wg_dot = sim.STG1_wg[2]
        wg_dtps = sim.STG1_wg[3]
        wg_de = sim.STG1_wg[4]
        
        # Normalisation vars.
        VS = ac.lb_USER[0]

        gmin = np.asin(sim.lb[4] / VS)
        g_dot_max = gmin**2 / (sim.dT*tF)
        de_max = ac.ub[8]
        
        for k in range(N-1):
            # Actual and next states and functions.
            wi = w[9*k:9*(k+1)]
            wj = w[9*(k+1):9*(k+2)]
            fi = NLP_STG1.DYNAMIC_EQUATIONS(wi,ac,at,sim)
            fj = NLP_STG1.DYNAMIC_EQUATIONS(wj,ac,at,sim)

            # Alpha computation.
            uai = wi[0] - sim.wind[0]
            uaj = wj[0] - sim.wind[0]
            wai = wi[1] - sim.wind[1]
            waj = wj[1] - sim.wind[1]

            ai = ca.atan2(wai,uai)
            aj = ca.atan2(waj,uaj)

            # Theta retrieving.
            thi = wi[3]
            thj = wj[3]

            # Gamma computation.
            gi = (thi - ai) / gmin
            gj = (thj - aj) / gmin

            # Gamma dot computation.
            gi_dot = (wi[2] - (fi[1]*wi[0] - fi[0]*wi[1]) / (wi[0]**2 + wi[1]**2))**2 / g_dot_max
            gj_dot = (wj[2] - (fj[1]*wj[0] - fj[0]*wj[1]) / (wj[0]**2 + wj[1]**2))**2 / g_dot_max

            # Control rates.
            dtpsi = wi[7]
            dtpsj = wj[7]
            dtps_dot = (dtpsj - dtpsi)**2 / (dT*tF)
            
            dei = wi[8]
            dej = wj[8]
            de_dot = (dej - dei)**2 / (de_max**2*dT*tF)

            # Running cost at instants i and j.
            Li = wg_g*(gi - 1)**2 + wg_dot*gi_dot + wg_dtps*dtps_dot + wg_de*de_dot
            Lj = wg_g*(gj - 1)**2 + wg_dot*gj_dot + wg_dtps*dtps_dot + wg_de*de_dot

            # COST FUNCTIONAL (Running Cost)
            J += (dT * tF)/2 * (Li + Lj)
        
        # COST FUNCTIONAL (Terminal Cost)
        J += wg_tF*tF
        return J

class NLP_STG2:
    # MISSION CRUISE STAGE. DISCHARGE
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
        opts['ipopt.max_iter'] = 3000
        opts['ipopt.tol'] = 1e-6
        opts['ipopt.acceptable_tol'] = 1e-6
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
        dtps_trim    = y_opt[1]
        de_trim    = y_opt[2]
        
        u = V * np.cos(alpha_trim)
        w = V * np.sin(alpha_trim)
        
        w0_state = np.array([u, w, 0.0, alpha_trim, 0.0, -h, m])
        w0_control = [dtps_trim, de_trim, sim.t0[1]]
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
        T_max = ca.if_else(x7 > (ac.BEM + ac.PM), T_max, 0.0)
        M_T_max = ca.if_else(x7 > (ac.BEM + ac.PM), M_T_max, 0.0)
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
        mk = ca.if_else(x7 > (ac.BEM + ac.PM), -ac.SFC,0.0) # SFC.
        mk += ca.if_else(ca.logic_and(x5 > (sim.Vtp*(sim.lb[1]*sim.f_tp - sim.td/2)),x5 <= (sim.Vtp*(sim.lb[1]*sim.f_tp + sim.td/2))), -ac.PM/sim.td, 0.0) # Mass discharge.
        
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
            href_l = -sim.ub[3] - 3.0
            href_u = -sim.ub[3] + 3.0
            hi = -wi[5]

            # Path constraints. Inequality constraints.
            g_path.append(href_l - hi)
            lbg_path.append(-1e20)
            ubg_path.append(0)
            g_path.append(hi - href_u)
            lbg_path.append(-1e20)
            ubg_path.append(0)   

        # DISCHARGE DISTANCE CONSTRAINT
        x0 = w[4]
        xf = w[9*N-5]

        # Path constraints. Inequality constraints.
        g_path.append(sim.Vtp*sim.lb[1] - xf + x0)
        lbg_path.append(-1e20)
        ubg_path.append(0)

        # Path constraint. Inequality constraints.
        g_path.append(xf - x0 - sim.Vtp*sim.ub[1])
        lbg_path.append(-1e20)
        ubg_path.append(0)   

        return g_path, lbg_path, ubg_path
    
    @staticmethod
    def INITIAL_STATE_CONSTRAINTS(w,w0):
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

        return g_0, lbg_0, ubg_0

    @staticmethod
    def SIMPLE_BOUNDS(lb,ub,sim,N):
        lbx = []
        ubx = []

        # Iteration to asign simple bounds to each state and control variable.
        for k in range(N):
            for j in range(9):
                lbx.append(lb[j])
                ubx.append(ub[j])
        lbx.append(sim.lb[1])
        ubx.append(sim.ub[1])

        return lbx, ubx     

    @staticmethod
    def CONSTRAINTS_AND_BOUNDS(x,u,utf,ac,at,sim):
        # Time step, number of nodes and state bounds.
        dT = sim.dT
        N = sim.N
        lb = ac.lb
        ub = ac.ub

        # STATE AND CONTROL VECTORS
        w = NLP_STG2.STATES_CONTROL_VECTOR(x,u,utf,N)
        tF = w[9*N]

        # Computation of trim condition.
        w0_trim = NLP_STG2.COMPUTE_TRIM(ac,at,sim)
        sim.w0 = w0_trim[:9]
        
        # Reconstruction of full initial guess
        # as planar functions.
        w0_ls = []
        for k in range(N):
            w0_ls.append(w0_trim[0])
            w0_ls.append(w0_trim[1])
            w0_ls.append(w0_trim[2])
            w0_ls.append(w0_trim[3])
            w0_ls.append(w0_trim[0] * np.cos(w0_trim[3]) * dT * k)
            w0_ls.append(w0_trim[5])
            w0_ls.append(w0_trim[6])
            w0_ls.append(w0_trim[7])
            w0_ls.append(w0_trim[8])
        w0_ls.append(ca.DM(sim.t0[1]))
        w0 = np.concatenate(w0_ls)

        # DYNAMIC COSNTRAINTS HANDLING
        g_dyn = []
        lbg_dyn = []
        ubg_dyn = []
        for k in range(N-1):
            wi = w[9*k:9*(k+1)]
            wj = w[9*(k+1):9*(k+2)]
            fi = NLP_STG2.DYNAMIC_EQUATIONS(wi,ac,at,sim)
            fj = NLP_STG2.DYNAMIC_EQUATIONS(wj,ac,at,sim)               
            g, lbg, ubg = NLP_STG2.DYNAMIC_CONSTRAINTS(wi[:7],wj[:7],fi[:7],fj[:7],dT,tF)
            g_dyn.append(g)
            lbg_dyn.append(lbg)
            ubg_dyn.append(ubg)
        g_dyn = sum(g_dyn,[])

        # PATH CONSTRAINTS HANDLING
        g_path, lbg_path, ubg_path = NLP_STG2.PATH_CONSTRAINTS(w,sim,N)

        # INITIAL STATE CONSTRAINTS HANDLING
        g_0, lbg_0, ubg_0 = NLP_STG2.INITIAL_STATE_CONSTRAINTS(w,sim.w0)

        # SIMPLE BOUNDS FOR STATES AND CONTROL
        lbx, ubx = NLP_STG2.SIMPLE_BOUNDS(lb,ub,sim,N)

        # Construction of final vectors.
        g = g_dyn + g_path + g_0
        lbg = lbg_dyn + lbg_path + lbg_0 
        ubg = ubg_dyn + ubg_path + ubg_0 

        return w0, w, lbx, ubx, g, lbg, ubg
    
    @staticmethod
    def COST_FUNCTIONAL(w,ac,at,sim):
        # Cost initialisation, time step and number
        # of nodes.
        J = 0
        dT = sim.dT
        N = sim.N
        tF = w[9*N]
        
        # Weights assignation.
        wg_tF = sim.STG2_wg[0]
        wg_g = sim.STG2_wg[1]
        wg_h = sim.STG2_wg[2]
        wg_g_dot = sim.STG2_wg[3]
        wg_dtps = sim.STG2_wg[4]
        wg_de = sim.STG2_wg[5]

        # Normalisation vars.
        VS = ac.lb_USER[0]
        gmax = np.asin(sim.ub[4] / VS)
        g_dot_max = gmax**2 / (dT*tF)
        de_max = ac.ub[8]
        href = -sim.ub[3]

        for k in range(N-1):
            # Actual and next states and functions.
            wi = w[9*k:9*(k+1)]
            wj = w[9*(k+1):9*(k+2)]
            fi = NLP_STG2.DYNAMIC_EQUATIONS(wi,ac,at,sim)
            fj = NLP_STG2.DYNAMIC_EQUATIONS(wj,ac,at,sim)

            # Alpha computation.
            uai = wi[0] - sim.wind[0]
            uaj = wj[0] - sim.wind[0]
            wai = wi[1] - sim.wind[1]
            waj = wj[1] - sim.wind[1]

            ai = ca.atan2(wai,uai)
            aj = ca.atan2(waj,uaj)

            # Theta retrieving.
            thi = wi[3]
            thj = wj[3]

            # Gamma computation.
            gi = (thi - ai)**2 / gmax**2
            gj = (thj - aj)**2 / gmax**2

             # Altitude computation.
            hi = -wi[5] / href
            hj = -wj[5] / href

            # Gamma dot computation.
            gi_dot = (wi[2] - (fi[1]*wi[0] - fi[0]*wi[1]) / (wi[0]**2 + wi[1]**2))**2 / g_dot_max
            gj_dot = (wj[2] - (fj[1]*wj[0] - fj[0]*wj[1]) / (wj[0]**2 + wj[1]**2))**2 / g_dot_max

            # Control rates.
            dtpsi = wi[7]
            dtpsj = wj[7]
            dtps_dot = (dtpsj - dtpsi)**2 / (dT*tF)
            
            dei = wi[8]
            dej = wj[8]
            de_dot = (dej - dei)**2 / (de_max**2*dT*tF)

            # Running cost at instants i and j.
            Li = wg_g*gi + wg_h*(hi - 1)**2 + wg_g_dot*gi_dot + wg_dtps*dtps_dot + wg_de*de_dot
            Lj = wg_g*gj + wg_h*(hj - 1)**2 + wg_g_dot*gj_dot + wg_dtps*dtps_dot + wg_de*de_dot

            # COST FUNCTIONAL (Running Cost)
            J += (dT*tF)/2 * (Li + Lj)
        
        # COST FUNCTIONAL (Terminal Cost)
        J += wg_tF*tF
        return J
    
class NLP_STG3:
    # MISSION CLIMB STAGE
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
        V = sim.x0_USER[0]  
        h = -sim.ub[3] 
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
        opts['ipopt.max_iter'] = 3000
        opts['ipopt.tol'] = 1e-6
        opts['ipopt.acceptable_tol'] = 1e-6
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
        w0_control = [dt_trim, de_trim, sim.t0[2]]
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
        T_max = ca.if_else(x7 > (ac.BEM + ac.PM), T_max, 0.0)
        M_T_max = ca.if_else(x7 > (ac.BEM + ac.PM), M_T_max, 0.0)
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
        mk = ca.if_else(x7 > (ac.BEM + ac.PM), -ac.SFC, 0.0) # SFC.
        
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
    def PATH_CONSTRAINTS(w,sim,ac,N):
        g_path = []
        lbg_path = []
        ubg_path = []

        for k in range(N-1):
            wi = w[9*k:9*(k+1)]
            # Aerodynamic velocity computation.
            ua = wi[0] - sim.wind[0]
            wa = wi[1] - sim.wind[1]
            V = ca.sqrt(ua**2 + wa**2)

            # VELOCITIES CONSTRAINTS
            VS = ac.lb_USER[0]
            VNO = ac.ub_USER[0]

            # Path constraints. Inequality constraints.
            g_path.append(VS - V)
            lbg_path.append(-1e20)
            ubg_path.append(0)
            g_path.append(V - VNO)
            lbg_path.append(-1e20)
            ubg_path.append(0)   

        # END TIME CONSTRAINT. Reference altitude and distance.
        href = -sim.lb[3]
        xref = sim.R_exit - (1 - sim.f_tp)*sim.Vtp*sim.lb[1]
        hf = -w[9*N-4]
        xf = w[9*N-5]
        x0 = w[4]

        # Path constraints. Equality constraint.
        g_path.append(hf - href)
        lbg_path.append(0)
        ubg_path.append(0)   

        # Path constraint. Inequality constraint.
        g_path.append((xf - x0) - xref)
        lbg_path.append(-1e20)
        ubg_path.append(0)   

        return g_path, lbg_path, ubg_path
    
    @staticmethod
    def INITIAL_STATE_CONSTRAINTS(w,w0):
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

        return g_0, lbg_0, ubg_0

    @staticmethod
    def SIMPLE_BOUNDS(lb,ub,sim,N):
        lbx = []
        ubx = []

        # Iteration to asign simple bounds to each state and control variable.
        for k in range(N):
            for j in range(9):
                lbx.append(lb[j])
                ubx.append(ub[j])
        lbx.append(sim.lb[2])
        ubx.append(sim.ub[2])

        return lbx, ubx     

    @staticmethod
    def CONSTRAINTS_AND_BOUNDS(x,u,utf,ac,at,sim):
        # Time step, number of nodes and state bounds.
        dT = sim.dT
        N = sim.N
        lb = ac.lb
        ub = ac.ub

        # STATE AND CONTROL VECTORS
        w = NLP_STG3.STATES_CONTROL_VECTOR(x,u,utf,N)
        tF = w[9*N]

        # Computation of trim condition.
        w0_trim = NLP_STG3.COMPUTE_TRIM(ac,at,sim)
        sim.w0 = w0_trim[:9]
        
        # Reconstruction of full initial guess
        # as planar functions.
        w0_ls = []
        for k in range(N):
            w0_ls.append(w0_trim[0])
            w0_ls.append(w0_trim[1])
            w0_ls.append(w0_trim[2])
            w0_ls.append(w0_trim[3])
            w0_ls.append(w0_trim[0] * np.cos(w0_trim[3]) * dT * k)
            w0_ls.append(w0_trim[5])
            w0_ls.append(w0_trim[6])
            w0_ls.append(w0_trim[7])
            w0_ls.append(w0_trim[8])
        w0_ls.append(ca.DM(sim.t0[2]))
        w0 = np.concatenate(w0_ls)

        # DYNAMIC COSNTRAINTS HANDLING
        g_dyn = []
        lbg_dyn = []
        ubg_dyn = []
        for k in range(N-1):
            wi = w[9*k:9*(k+1)]
            wj = w[9*(k+1):9*(k+2)]
            fi = NLP_STG3.DYNAMIC_EQUATIONS(wi,ac,at,sim)
            fj = NLP_STG3.DYNAMIC_EQUATIONS(wj,ac,at,sim)               
            g, lbg, ubg = NLP_STG3.DYNAMIC_CONSTRAINTS(wi[:7],wj[:7],fi[:7],fj[:7],dT,tF)
            g_dyn.append(g)
            lbg_dyn.append(lbg)
            ubg_dyn.append(ubg)
        g_dyn = sum(g_dyn,[])

        # PATH CONSTRAINTS HANDLING
        g_path, lbg_path, ubg_path = NLP_STG3.PATH_CONSTRAINTS(w,sim,ac,N)

        # INITIAL AND FINAL STATE CONSTRAINTS HANDLING
        g_0, lbg_0, ubg_0 = NLP_STG3.INITIAL_STATE_CONSTRAINTS(w,sim.w0)

        # SIMPLE BOUNDS FOR STATES AND CONTROL
        lbx, ubx = NLP_STG3.SIMPLE_BOUNDS(lb,ub,sim,N)

        # Construction of final vectors.
        g = g_dyn + g_path + g_0
        lbg = lbg_dyn + lbg_path + lbg_0
        ubg = ubg_dyn + ubg_path + ubg_0

        return w0, w, lbx, ubx, g, lbg, ubg
    
    @staticmethod
    def COST_FUNCTIONAL(w,ac,at,sim):
        # Cost initialisation, time step and number
        # of nodes.
        J = 0
        dT = sim.dT
        N = sim.N
        tF = w[9*N]

        # Weights assignation.
        wg_tF = sim.STG3_wg[0]
        wg_g_dot = sim.STG3_wg[1]
        wg_sp = sim.STG3_wg[2] 
        wg_dtps = sim.STG3_wg[3]
        wg_de = sim.STG3_wg[4]
        wg_de_sat = sim.STG3_wg[5]
        
        # Normalisation vars.
        VS = ac.lb_USER[0]
        gmax = np.asin(sim.ub[4] / VS)
        g_dot_max = gmax**2 / (dT*tF)
        asf = 0.75*ac.ub[3]
        amg = 0.25*ac.ub[3]
        de_max = ac.ub[8]
        
        for k in range(N-1):
            # Actual and next states and functions.
            wi = w[9*k:9*(k+1)]
            wj = w[9*(k+1):9*(k+2)]
            fi = NLP_STG3.DYNAMIC_EQUATIONS(wi,ac,at,sim)
            fj = NLP_STG3.DYNAMIC_EQUATIONS(wj,ac,at,sim)

            # Alpha computation.
            uai = wi[0] - sim.wind[0]
            uaj = wj[0] - sim.wind[0]
            wai = wi[1] - sim.wind[1]
            waj = wj[1] - sim.wind[1]

            ai = ca.atan2(wai,uai)
            aj = ca.atan2(waj,uaj)

            # Stall protection.
            spi = ca.mmax(ca.if_else((ai - asf) > 0.0, ai - asf, 0.0))**2 / amg**2
            spj = ca.mmax(ca.if_else((aj - asf) > 0.0, ai - asf, 0.0))**2 / amg**2

            # Gamma dot computation.
            gi_dot = (wi[2] - (fi[1]*wi[0] - fi[0]*wi[1]) / (wi[0]**2 + wi[1]**2))**2 / g_dot_max
            gj_dot = (wj[2] - (fj[1]*wj[0] - fj[0]*wj[1]) / (wj[0]**2 + wj[1]**2))**2 / g_dot_max

            # Control rates.
            dtpsi = wi[7]
            dtpsj = wj[7]
            dtps_dot = (dtpsj - dtpsi)**2 / (dT*tF)
            
            dei = wi[8]
            dej = wj[8]
            de_dot = (dej - dei)**2 / (de_max**2*dT*tF)

            # Elevator saturation.
            desi = dei**4 / de_max**4
            desj = dej**4 / de_max**4

            # Running cost at instants i and j.
            Li = wg_g_dot*gi_dot + wg_sp*spi + wg_dtps*dtps_dot + wg_de*de_dot + wg_de_sat*desi
            Lj = wg_g_dot*gj_dot + wg_sp*spj + wg_dtps*dtps_dot + wg_de*de_dot + wg_de_sat*desj

            # COST FUNCTIONAL (Running Cost)
            J += (dT*tF)/2 * (Li + Lj)
        
        # COST FUNCTIONAL (Terminal Cost)
        J += wg_tF*tF
        return J