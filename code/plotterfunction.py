import matplotlib as mpl
import matplotlib.pyplot as plt
import os

class Plotter:
    def GENERATE_PLOT(x,y,varsname,titles,path):
        # Create figure and assign its size.
        plt.figure(figsize=(9,6))

        # Define greyscale colours and linestyles.
        linestyles = ['-', '--', '-.']    
        colors = [(0,0,0), (0.25,0.25,0.25), (0.64,0.64,0.64)]

        if y.ndim == 1:
            # Unique y set of vars.
            vars = 1
            plt.plot(x,y, label=varsname,color = colors[0], linestyle = linestyles[0], linewidth=1.5)
        else:
            # Multiple y sets of vars.
            vars = y.shape[1]
            for k in range(vars):
                # Plot all vars with the same x-axis.
                plt.plot(x,y[:,k], label = varsname[k], color = colors[k], linestyle = linestyles[k], linewidth=1.5)

        # Add titles and legends.
        plt.xlabel(titles[0], fontsize = 14, fontstyle='italic', fontfamily='serif')
        plt.ylabel(titles[1], fontsize = 14, fontstyle='italic', fontfamily='serif')
        plt.title(titles[2], fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
        plt.legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.grid(True, which='major', linestyle='--', linewidth=0.8, color='gray', alpha=0.7)
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
        plt.tight_layout()

        # Save to indicated path.
        plt.savefig(os.path.join(path, titles[3]))
        plt.close()

        
