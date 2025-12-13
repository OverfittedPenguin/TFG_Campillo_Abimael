import matplotlib.pyplot as plt
import numpy as np
import os

class Plotter:
    def GENERATE_MANOEUVRE_TRAJECTORIES(t,stg1,stg2,stg3,ac,sim,path):
        colors_hex = ['#001233', '#003874', '#007FFF',  "#4DB6AC", '#A9B7C7', '#ef233c']
        # States and controls storage vectors.
        x1_1 = []
        x2_1 = []
        x3_1 = []
        x4_1 = []
        x5_1 = []
        x6_1 = []
        x7_1 = []
        u1_1 = []
        u2_1 = []

        x1_2 = []
        x2_2 = []
        x3_2 = []
        x4_2 = []
        x5_2 = []
        x6_2 = []
        x7_2 = []
        u1_2 = []
        u2_2 = []

        x1_3 = []
        x2_3 = []
        x3_3 = []
        x4_3 = []
        x5_3 = []
        x6_3 = []
        x7_3 = []
        u1_3 = []
        u2_3 = []

        V = []
        alpha = []
        g = []

        for k in range(sim.N):
            # For each stage and variable, the current value is retrieved and 
            # appended into the respective storage array.
            idx = 9*k

            x1_1.append(stg1[idx])
            x2_1.append(stg1[idx + 1])
            x3_1.append(stg1[idx + 2])
            x4_1.append(stg1[idx + 3])
            x5_1.append(stg1[idx + 4])
            x6_1.append(stg1[idx + 5])
            x7_1.append(stg1[idx + 6])
            u1_1.append(stg1[idx + 7])
            u2_1.append(stg1[idx + 8])

            x1_2.append(stg2[idx])
            x2_2.append(stg2[idx + 1])
            x3_2.append(stg2[idx + 2])
            x4_2.append(stg2[idx + 3])
            x6_2.append(stg2[idx + 5])
            x7_2.append(stg2[idx + 6])
            u1_2.append(stg2[idx + 7])
            u2_2.append(stg2[idx + 8])

            x1_3.append(stg3[idx])
            x2_3.append(stg3[idx + 1])
            x3_3.append(stg3[idx + 2])
            x4_3.append(stg3[idx + 3])
            x6_3.append(stg3[idx + 5])
            x7_3.append(stg3[idx + 6])
            u1_3.append(stg3[idx + 7])
            u2_3.append(stg3[idx + 8])

        for k in range(sim.N):
            # Horizontal distance corrected (from relative to absolute).
            idx = 9*k
            x5_2.append(stg2[idx + 4] + x5_1[sim.N-1])

        for k in range(sim.N):
            # Horizontal distance corrected (from relative to absolute).
            idx = 9*k
            x5_3.append(stg3[idx + 4] + x5_2[sim.N-1])

        # Arrays append for full manoeuvre representation.
        x1 = np.concatenate([x1_1, x1_2, x1_3])
        x2 = np.concatenate([x2_1, x2_2, x2_3])
        x3 = np.concatenate([x3_1, x3_2, x3_3])
        x4 = np.concatenate([x4_1, x4_2, x4_3])
        x5 = np.concatenate([x5_1, x5_2, x5_3])
        x6 = np.concatenate([x6_1, x6_2, x6_3])
        x7 = np.concatenate([x7_1, x7_2, x7_3])
        u1 = np.concatenate([u1_1, u1_2, u1_3])
        u2 = np.concatenate([u2_1, u2_2, u2_3])

        # Arrays conversion.
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)
        x4 = np.array(x4)
        x5 = np.array(x5)
        x6 = np.array(x6)
        x7 = np.array(x7)
        u1 = np.array(u1)
        u2 = np.array(u2)

        # Arrays conversion.
        x1 = np.squeeze(x1)
        x2 = np.squeeze(x2)
        x3 = np.squeeze(x3)
        x4 = np.squeeze(x4)
        x5 = np.squeeze(x5)
        x6 = np.squeeze(x6)
        x7 = np.squeeze(x7)
        u1 = np.squeeze(u1)
        u2 = np.squeeze(u2)

        for k in range(len(x1)):
            # Computation of aerodynamic velocity, angle of attack
            # and flight path angle.
            ua, wa = x1[k] - sim.wind[0], x2[k] - sim.wind[1]
            alphai = np.arctan2(wa, ua)
            Vi = np.sqrt(ua**2 + wa**2)
            gi = x4[k] - alphai
            V.append(Vi)
            alpha.append(alphai)
            g.append(gi)

        # Arrays conversion.
        V = np.array(V)
        alpha = np.array(alpha)
        g = np.array(g)

        # PLOT 1: STATES
        fig1, axs = plt.subplots(
            nrows=2, 
            ncols=2, 
            figsize=(10,8),
            gridspec_kw={'hspace': 0.30, 'wspace': 0.70}
            )
        fig1.subplots_adjust(
            left=0.15,  
            right=0.85, 
            )

        # VELOCITIES
        axs[0,0].plot(t,x1,label="u",color=colors_hex[4],linestyle="--",linewidth=1.5)
        axs[0,0].plot(t,x2,label="w",color=colors_hex[2],linestyle="-.",linewidth=1.5)
        axs[0,0].plot(t,V,label="V",color=colors_hex[0],linestyle="-",linewidth=1.2)

        # Bounds.
        V_NO = np.ones(len(V)) * ac.ub_USER[0]
        V_S = np.ones(len(V)) * ac.lb_USER[0]
        axs[0,0].plot(t,V_NO,label=r"$V_{NO}$",color=colors_hex[5],linestyle="--",linewidth=1.5)
        axs[0,0].plot(t,V_S,label=r"$V_{S}$",color=colors_hex[5],linestyle="-.",linewidth=1.5)

        # Titles, grid and legend.
        axs[0,0].set_xlabel("Time [s]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[0,0].set_ylabel("Velocity [m/s]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[0,0].set_title("Velocities",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[0,0].minorticks_on()
        axs[0,0].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[0,0].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))

        # ANGLES
        axs[0,1].plot(t,x4,label=r"$\theta$",color=colors_hex[3],linestyle="--",linewidth=1.5)
        axs[0,1].plot(t,alpha,label=r"$\alpha$",color=colors_hex[2],linestyle="-.",linewidth=1.5)
        axs[0,1].plot(t,g,label=r"$\gamma$",color=colors_hex[1],linestyle="-",linewidth=1.5)

        # Bounds.
        th_max= np.ones(len(alpha))*ac.ub[3]
        th_min = np.ones(len(alpha))*ac.lb[3]
        astall = np.concatenate([np.ones(sim.N*2)*ac.ub[3], np.ones(sim.N)*ac.ub[3]*0.6])
        axs[0,1].plot(t,th_max,label=r"$\theta_{ub}$",color=colors_hex[5],linestyle="--",linewidth=1.5)
        axs[0,1].plot(t,th_min,label=r"$\theta_{lb}$",color=colors_hex[5],linestyle="-.",linewidth=1.5)
        axs[0,1].plot(t,astall,label=r"$\alpha_{stall}$",color=colors_hex[0],linestyle="-.",linewidth=1.2)

        # Titles, grid and legend.
        axs[0,1].set_xlabel("Time [s]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[0,1].set_ylabel("Angle [rad]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[0,1].set_title("Angles",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[0,1].minorticks_on()
        axs[0,1].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[0,1].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))

        # MASS
        axs[1,0].plot(t,x7,label=r"m",color=colors_hex[0],linestyle="-",linewidth=1.5)

        # Bounds.
        m_max = np.ones(len(x7))*ac.ub[6]
        m_min = np.ones(len(x7))*ac.lb[6]
        axs[1,0].plot(t,m_max,label=r"$MTOM$",color=colors_hex[5],linestyle="--",linewidth=1.5)
        axs[1,0].plot(t,m_min,label=r"$BEM$",color=colors_hex[5],linestyle="-.",linewidth=1.5)

        # Titles, grid and legend.
        axs[1,0].set_xlabel("Time [s]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[1,0].set_ylabel("Mass [kg]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[1,0].set_title("Mass",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[1,0].minorticks_on()
        axs[1,0].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[1,0].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))

        # TRAJECTORY
        axs[1,1].plot(x5,-x6,label="Trajectory",color=colors_hex[0],linestyle="-",linewidth=1.5)

        # Bounds.
        h_max = np.ones(len(x5))*ac.ub[5]
        h_min = np.ones(len(x5))*ac.lb[5]
        h_ref = np.ones(len(x5))*sim.ub[3]
        h_cruise = np.ones(len(x5))*sim.lb[3]
        axs[1,1].plot(x5,-h_max,label=r"$h_{lb}$",color=colors_hex[5],linestyle="-.",linewidth=1.5)
        axs[1,1].plot(x5,-h_min,label=r"$h_{ub}$",color=colors_hex[5],linestyle="--",linewidth=1.5)
        axs[1,1].plot(x5,-h_cruise,label=r"$h_{cruise}$",color=colors_hex[3],linestyle="--",linewidth=1)
        axs[1,1].plot(x5,-h_ref,label=r"$h_{ref}$",color=colors_hex[3],linestyle="-.",linewidth=1)

        # Titles, grid and legend.
        axs[1,1].set_xlabel("Horizontal distance [m]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[1,1].set_ylabel("Altitude ASL [m]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[1,1].set_title("UAV trajectory",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[1,1].minorticks_on()
        axs[1,1].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[1,1].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))

        fig1.savefig(os.path.join(path, "STATES.svg"))

        # PLOT 2: CONTROLS
        fig2, axs = plt.subplots(
            nrows=1, 
            ncols=2, 
            figsize=(10,6),
            gridspec_kw={'hspace': 0.30, 'wspace': 0.70}
            )
        fig2.subplots_adjust(
            left=0.10, 
            bottom=0.25, 
            right=0.85, 
            top=0.80
            )

        # TPS
        axs[0].plot(t,u1,label="TPS",color=colors_hex[0],linestyle="-",linewidth=1.5)

        # Bounds.
        TPS_max = np.ones(len(t))*ac.ub[7]
        TPS_min = np.zeros(len(t))
        axs[0].plot(t,TPS_max,label=r"$TPS_{ub}$",color=colors_hex[5],linestyle="--",linewidth=1.5)
        axs[0].plot(t,TPS_min,label=r"$TPS_{lb}$",color=colors_hex[5],linestyle="-.",linewidth=1.5)

        # Titles, grid and legend.
        axs[0].set_xlabel("Time [s]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[0].set_ylabel("TPS - throttle position sensor [-]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[0].set_title(r"$\delta_{TPS}$ control",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[0].minorticks_on()
        axs[0].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[0].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))

        # ELEVATOR
        axs[1].plot(t,u2,label=r"$\delta_e$",color=colors_hex[0],linestyle="-",linewidth=1.5)

        # Bounds.
        de_max= np.ones(len(t))*ac.ub[8]
        de_min = np.ones(len(t))*ac.lb[8]
        axs[1].plot(t,de_max,label=r"$\delta_{e,ub}$",color=colors_hex[5],linestyle="--",linewidth=1.5)
        axs[1].plot(t,de_min,label=r"$\delta_{e,lb}$",color=colors_hex[5],linestyle="-.",linewidth=1.5)

        # Titles, grid and legend.
        axs[1].set_xlabel("Time [s]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[1].set_ylabel(r"$\delta_e$ - elevator deflection [rad]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[1].set_title(r"$\delta_e$ control",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[1].minorticks_on()
        axs[1].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[1].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))
        fig2.savefig(os.path.join(path, "CONTROLS.svg"))
        plt.show()
    
    def GENERATE_RESULTS_PLOT(t,x,x0,ac,sim,path):
        colors_hex = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#595959']
        # States and controls storage vectors.
        x0_1 = []
        x0_2 = []
        x0_3 = []
        x0_4 = []
        x0_5 = []
        x0_6 = []
        x0_7 = []
        u0_1 = []
        u0_2 = []
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        x5 = []
        x6 = []
        x7 = []
        u1 = []
        u2 = []
        V = []
        alpha = []
        g = []

        for k in range(sim.N):
            # For each variable, the current value is retrieved and 
            # appended into the respective storage array.
            idx = 9*k
            x0_1.append(x0[idx])
            x0_2.append(x0[idx + 1])
            x0_3.append(x0[idx + 2])
            x0_4.append(x0[idx + 3])
            x0_5.append(x0[idx + 4])
            x0_6.append(x0[idx + 5])
            x0_7.append(x0[idx + 6])
            u0_1.append(x0[idx + 7])
            u0_2.append(x0[idx + 8])

            x1.append(x[idx])
            x2.append(x[idx + 1])
            x3.append(x[idx + 2])
            x4.append(x[idx + 3])
            x5.append(x[idx + 4])
            x6.append(x[idx + 5])
            x7.append(x[idx + 6])
            u1.append(x[idx + 7])
            u2.append(x[idx + 8])

        # Arrays conversion.
        x0_1 = np.array(x0_1)
        x0_2 = np.array(x0_2)
        x0_3 = np.array(x0_3)
        x0_4 = np.array(x0_4)
        x0_5 = np.array(x0_5)
        x0_6 = np.array(x0_6)
        x0_7 = np.array(x0_7)
        u0_1 = np.array(u0_1)
        u0_2 = np.array(u0_2)
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)
        x4 = np.array(x4)
        x5 = np.array(x5)
        x6 = np.array(x6)
        x7 = np.array(x7)
        u1 = np.array(u1)
        u2 = np.array(u2)

                # Arrays conversion.
        x0_1 = np.squeeze(x0_1)
        x0_2 = np.squeeze(x0_2)
        x0_3 = np.squeeze(x0_3)
        x0_4 = np.squeeze(x0_4)
        x0_5 = np.squeeze(x0_5)
        x0_6 = np.squeeze(x0_6)
        x0_7 = np.squeeze(x0_7)
        u0_1 = np.squeeze(u0_1)
        u0_2 = np.squeeze(u0_2)
        x1 = np.squeeze(x1)
        x2 = np.squeeze(x2)
        x3 = np.squeeze(x3)
        x4 = np.squeeze(x4)
        x5 = np.squeeze(x5)
        x6 = np.squeeze(x6)
        x7 = np.squeeze(x7)
        u1 = np.squeeze(u1)
        u2 = np.squeeze(u2)

        for k in range(len(x1)):
            # Computation of aerodynamic velocity, angle of attack
            # and flight path angle.
            ua, wa = x1[k] - sim.wind[0], x2[k] - sim.wind[1]
            alphai = np.arctan2(wa, ua)
            Vi = np.sqrt(ua**2 + wa**2)
            gi = x4[k] - alphai
            V.append(Vi)
            alpha.append(alphai)
            g.append(gi)

        # Arrays conversion.
        V = np.array(V)
        Vtp = np.ones(len(V))*sim.Vtp
        alpha = np.array(alpha)
        g = np.array(g)

        # PLOT 1: STATES
        fig1, axs = plt.subplots(
            nrows=2, 
            ncols=2, 
            figsize=(10,8),
            gridspec_kw={'hspace': 0.30, 'wspace': 0.70}
            )
        fig1.subplots_adjust(
            left=0.15,  
            right=0.85, 
            )

        # VELOCITIES
        axs[0,0].plot(t,x1,label="u",color=colors_hex[0],linestyle="--",linewidth=1.5)
        axs[0,0].plot(t,x2,label="w",color=colors_hex[2],linestyle="-.",linewidth=1.5)
        axs[0,0].plot(t,V,label="V",color=colors_hex[4],linestyle=":",linewidth=1.5)
        axs[0,0].plot(t,Vtp,label=r"$V_{tp}$",color="black",linestyle="-",linewidth=1.5)

        # Bounds.
        V_max = np.ones(len(V))*np.sqrt(ac.ub[0]**2 + ac.ub[1]**2)
        V_min = np.zeros(len(V))
        axs[0,0].plot(t,V_max,label=r"$V_{ub}$",color="red",linestyle="--",linewidth=1.5)
        axs[0,0].plot(t,V_min,label=r"$V_{lb}$",color="red",linestyle="--",linewidth=1.5)

        # Titles, grid and legend.
        axs[0,0].set_xlabel("Time [s]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[0,0].set_ylabel("Velocity [m/s]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[0,0].set_title("Velocities through time",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[0,0].minorticks_on()
        axs[0,0].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[0,0].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))

        # ANGLES
        axs[0,1].plot(t,x4,label=r"$\theta$",color=colors_hex[0],linestyle="--",linewidth=1.5)
        axs[0,1].plot(t,alpha,label=r"$\alpha$",color=colors_hex[4],linestyle="-.",linewidth=1.5)
        axs[0,1].plot(t,g,label=r"$\gamma$",color=colors_hex[5],linestyle="-",linewidth=1.5)

        # Bounds.
        th_max= np.ones(len(alpha))*ac.ub[3]
        th_min = np.ones(len(alpha))*ac.lb[3]
        axs[0,1].plot(t,th_max,label=r"$\theta_{ub}$",color="red",linestyle="--",linewidth=1.5)
        axs[0,1].plot(t,th_min,label=r"$\theta_{lb}$",color="red",linestyle="--",linewidth=1.5)

        # Titles, grid and legend.
        axs[0,1].set_xlabel("Time [s]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[0,1].set_ylabel("Angle [rad]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[0,1].set_title("Angles through time",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[0,1].minorticks_on()
        axs[0,1].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[0,1].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))

        # MASS
        axs[1,0].plot(t,x7,label=r"m",color=colors_hex[0],linestyle="-",linewidth=1.5)

        # Bounds.
        m_max = np.ones(len(x7))*ac.ub[6]
        m_min = np.ones(len(x7))*ac.lb[6]
        axs[1,0].plot(t,m_max,label=r"$m_{ub}$",color="red",linestyle="--",linewidth=1.5)
        axs[1,0].plot(t,m_min,label=r"$m_{lb}$",color="red",linestyle="--",linewidth=1.5)

        # Titles, grid and legend.
        axs[1,0].set_xlabel("Time [s]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[1,0].set_ylabel("Mass [kg]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[1,0].set_title("Mass through time",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[1,0].minorticks_on()
        axs[1,0].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[1,0].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))

        # TRAJECTORY
        axs[1,1].plot(x5,-x6,label="Trajectory",color=colors_hex[0],linestyle="-",linewidth=1.5)

        # Bounds.
        h_max = np.ones(len(x5))*ac.ub[5]
        h_min = np.ones(len(x5))*ac.lb[5]
        axs[1,1].plot(x5,-h_max,label=r"$h_{ub}$",color="red",linestyle="--",linewidth=1.5)
        axs[1,1].plot(x5,-h_min,label=r"$h_{lb}$",color="red",linestyle="--",linewidth=1.5)

        # Titles, grid and legend.
        axs[1,1].set_xlabel("Horizontal distance [m]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[1,1].set_ylabel("Altitude ASL [m]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[1,1].set_title("UAV trajectory",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[1,1].minorticks_on()
        axs[1,1].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[1,1].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))

        fig1.savefig(os.path.join(path, "STATES.svg"))

        # PLOT 2: CONTROLS
        fig2, axs = plt.subplots(
            nrows=1, 
            ncols=2, 
            figsize=(10,6),
            gridspec_kw={'hspace': 0.30, 'wspace': 0.70}
            )
        fig2.subplots_adjust(
            left=0.10, 
            bottom=0.25, 
            right=0.85, 
            top=0.80
            )

        # TPS
        axs[0].plot(t,u1,label="TPS",color="black",linestyle="-",linewidth=1.5)
        axs[0].plot(t,u0_1,label=r"$TPS_0$",color="grey",linestyle="--",linewidth=1.5)

        # Bounds.
        TPS_max = np.ones(len(t))*ac.ub[7]
        TPS_min = np.zeros(len(t))
        axs[0].plot(t,TPS_max,label=r"$TPS_{ub}$",color="red",linestyle="--",linewidth=1.5)
        axs[0].plot(t,TPS_min,label=r"$TPS_{lb}$",color="red",linestyle="--",linewidth=1.5)

        # Titles, grid and legend.
        axs[0].set_xlabel("Time [s]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[0].set_ylabel("TPS - throttle position sensor [-]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[0].set_title("TPS through time",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[0].minorticks_on()
        axs[0].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[0].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))

        # ELEVATOR
        axs[1].plot(t,u2,label=r"$\delta_e$",color="black",linestyle="-",linewidth=1.5)
        axs[1].plot(t,u0_2,label=r"$\delta_{e,0}$",color="grey",linestyle="--",linewidth=1.5)

        # Bounds.
        de_max= np.ones(len(t))*ac.ub[8]
        de_min = np.ones(len(t))*ac.lb[8]
        axs[1].plot(t,de_max,label=r"$\delta_{e,ub}$",color="red",linestyle="--",linewidth=1.5)
        axs[1].plot(t,de_min,label=r"$\delta_{e,lb}$",color="red",linestyle="--",linewidth=1.5)

        # Titles, grid and legend.
        axs[1].set_xlabel("Time [s]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        axs[1].set_ylabel(r"$\delta_e$ - elevator deflection [rad]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        axs[1].set_title(r"$\delta_e$ through time",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        axs[1].minorticks_on()
        axs[1].grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        axs[1].legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))
        fig2.savefig(os.path.join(path, "CONTROLS.svg"))
        plt.show()

    def GENERATE_COST_PLOT(it,obj,time,path):
        # PLOT: COST EVOLUTION
        plt.figure(figsize=(12,8))
        plt.subplots_adjust(
            left=0.10, 
            bottom=0.20, 
            right=0.85, 
            top=0.85
            )
        
        # Cost plot.
        plt.plot(it,obj,label="Cost J",color='#004D99',linestyle="-",linewidth=1.5)

        # Titles, grid and legend.
        plt.xlabel("Iterations [-]",fontsize = 14,fontstyle='italic',fontfamily='serif')
        plt.ylabel("Cost objective [-]",fontsize = 14, fontstyle='italic', fontfamily='serif')
        plt.title(f"Objective evolution per iteration. Computation time: {time}s",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
        plt.legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))
        plt.savefig(os.path.join(path, "COST.svg"))
        plt.show()

