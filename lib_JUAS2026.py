#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import matplotlib.animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def animation_plot(particle1): 

    points=[(particle1.x[0,0],particle1.px[0,0]),(particle1.x[0,1],particle1.px[0,1]),(particle1.x[0,2],particle1.px[0,2]),(particle1.x[0,3],particle1.px[0,3]),(particle1.x[0,4],particle1.px[0,4]),(particle1.x[0,5],particle1.px[0,5]),(particle1.x[0,6],particle1.px[0,6])]

    x_data=particle1.x[0:7]
    y_data=particle1.px[0:7]

    fig, ax = plt.subplots()
    #ax.set_xlim(min(x_data)*2, max(x_data)*2)  
    #ax.set_ylim(min(y_data)*2, max(y_data)*2)  
    ax.set_xlabel("x [m]")
    ax.set_ylabel("px")
    ax.legend(loc="upper left") 

    # Adding phase space ellipse on the plot
    # Twiss parameters
    beta = 170     # Beta Twiss
    alpha = -2.4   # Alpha Twiss
    gamma = (1 + alpha**2) / beta  # Gamma Twiss
    epsilon = 0.001**2/beta+alpha**2*0.001**2/beta  # Emittance corresponding to the initial coordinates

    # Points of the ellipse
    theta = np.linspace(0, 2 * np.pi, 500)  # Parametric angle to build the ellipse
    xa = np.sqrt(epsilon * beta) * np.cos(theta)  # x
    xpa = -np.sqrt(epsilon / beta) * (alpha * np.cos(theta) + np.sin(theta))  # x'

    # Configuración de la gráfica
    ax.plot(xa, xpa, label="Elipse en el espacio de fase", color="blue")

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
    shapes = ['o', '^', 's', 'P', 'D', '*', 'h']  # Circle, triangle, square, etc.

    # Initialize plot elements
    point, = ax.plot([], [], 'ro', markersize=10, label="Current Point")  # Current point
    trail = []  # Store the trail of points as individual plots

    # Initialization function
    def init():
        point.set_data([], [])
        return point,

    # Update function for each frame
    def update(frame):
        # Update the current point
        x, y = points[frame]
        point.set_data([x], [y])
        point.set_color(colors[frame % len(colors)])  # Change color dynamically
        point.set_marker(shapes[frame % len(shapes)])  # Change shape dynamically

        # Add the current point to the trail
        trail_point, = ax.plot(x, y, marker=shapes[frame % len(shapes)], color=colors[frame % len(colors)], markersize=8)
        trail.append(trail_point)

        
        ax.legend(
            labels=[
                f"Phase space ellipse" ,  
                f"Turn number {frame} "  
            ],
            loc="upper left"
        )

        return [point] + trail  # Return all elements to re-render

    # Create the animation
    frames = len(points)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=500)

    return ani
    
def plot_phase_space_bunch(p_in, p_out, nbins=100, scale_mm=True):
    """
    Plot phase space distributions at entrance and exit.

    Parameters
    ----------
    p_in : Particles
        Particles at entrance
    p_out : Particles
        Particles at exit
    nbins : int
        Number of bins for hist2d
    scale_mm : bool
        If True, convert x,y to mm
    """

    scale = 1e3 if scale_mm else 1.0
    unit = 'mm' if scale_mm else 'm'

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    plots = [
        (p_in.x*scale,  p_in.px,  "Entrance: x vs px", "x",  "px"),
        (p_out.x*scale, p_out.px, "Exit: x vs px",     "x",  "px"),
        (p_in.y*scale,  p_in.py,  "Entrance: y vs py", "y",  "py"),
        (p_out.y*scale, p_out.py, "Exit: y vs py",     "y",  "py"),
        (p_in.x*scale,  p_in.y*scale,  "Entrance: x vs y", "x",  "y"),
        (p_out.x*scale, p_out.y*scale, "Exit: x vs y",     "x",  "y"),
    ]

    for ax, (X, Y, title, xlabel, ylabel) in zip(axes.flatten(), plots):
        h = ax.hist2d(X, Y, bins=nbins)
        fig.colorbar(h[3], ax=ax)
        ax.set_title(title)
        ax.set_xlabel(f"{xlabel} [{unit}]")
        ax.set_ylabel(f"{ylabel}")
        

    plt.tight_layout()
    plt.show()

def track_and_plot_bunch(data, part,tw, plane):
    plt.figure() 
    ax = plt.gca() 
    bsplt = tw.plot(lattice_only=True, ax=ax) 
    ax = bsplt.left
    if plane=='Horizontal':    
        ax.plot(data.s.T, data.x.T, '-')
        ax.set_xlabel('s [m]')
        ax.set_ylabel('x [m]')
    if plane=='Vertical':
        ax.plot(data.s.T, data.y.T, '-')
        ax.set_xlabel('s [m]')
        ax.set_ylabel('y [m]') 
    bsplt.lattice.set_ylim(-5, 1) 
    plt.show()
    return

def phase_space_plot(PartData1, plane1, title1, PartData2=None, plane2=None, title2=None):
    """
    Plot phase space for one or two particle datasets.

    Parameters
    ----------
    PartData1 : object
        First particle data (with attributes x, px, y, py)
    plane1 : str
        'Horizontal' or 'Vertical' for first plot
    PartData2 : object, optional
        Second particle data
    plane2 : str, optional
        'Horizontal' or 'Vertical' for second plot
    """

    def get_data(PartData, plane):
        if plane == 'Horizontal':
            return PartData.x[0, :], PartData.px[0, :], 'x [m]', 'px'
        elif plane == 'Vertical':
            return PartData.y[0, :], PartData.py[0, :], 'y [m]', 'py'
        else:
            raise ValueError('plane must be "Horizontal" or "Vertical"')

    # Decide number of subplots
    if PartData2 is None:
        fig, ax = plt.subplots(1, 1, figsize=(10,4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10,4))

    # --- First plot ---
    x, y, xlabel, ylabel = get_data(PartData1, plane1)
    axes[0].scatter(x, y, s=10, color='r')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title1)
    

    # --- Second plot (optional) ---
    if PartData2 is not None:
        x2, y2, xlabel2, ylabel2 = get_data(PartData2, plane2)
        axes[1].scatter(x2, y2, s=10,color='b')
        axes[1].set_xlabel(xlabel2)
        axes[1].set_ylabel(ylabel2)
        axes[1].set_title(title2)
       
    plt.tight_layout()
    plt.show()
    return

def plot_amplitude_vs_turn(plane,
                           p1, label1='Particle 1',
                           p2=None, label2='Particle 2',
                           p3=None, label3='Particle 3', p4=None, label4='Particle 4'):
    """
    Plot amplitude vs turn number for up to three particles.

    Parameters
    ----------
    plane : str
        'Horizontal' or 'Vertical'
    p1, p2, p3 : particle objects
        Particle datasets (e.g. from Xsuite tracking)
    label1, label2, label3 : str
        Labels for legend
    """

    plt.rcParams['figure.dpi'] = 100
    plt.figure(figsize=(10,6))

    # Select plane
    if plane.lower().startswith('h'):
        coord = 'x'
        ylabel = 'x [m]'
    elif plane.lower().startswith('v'):
        coord = 'y'
        ylabel = 'y [m]'
    else:
        raise ValueError("plane must be 'Horizontal' or 'Vertical'")

    # Helper to extract data
    def get_data(p):
        return getattr(p, coord)[0, :]

    # Plot particles if provided
    if p1 is not None:
        plt.plot(get_data(p1), '.-', label=label1)
    if p2 is not None:
        plt.plot(get_data(p2), '.-', label=label2)
    if p3 is not None:
        plt.plot(get_data(p3), '.-', label=label3)
    if p4 is not None:
        plt.plot(get_data(p4), '.-', label=label4)

    plt.xlabel('Turn')
    plt.ylabel(ylabel)
    plt.title(f'{plane} amplitude vs turn number')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return