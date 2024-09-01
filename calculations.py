# Get data from .txt

data = np.genfromtxt(r'C:\Users\IL MIO PC\Desktop\DAVIDE\pythonmat\EXP 3\send\europiodata.txt')
x=data[:, 0]
y=data[:, 1]
err=data[:, 2]
b=data[:, 3]

# Fit from formula
from scipy.optimize import curve_fit
def exponential_decay(x, A, tau, c):
    return A  * np.exp((x) * tau)+c
params, cov = curve_fit(exponential_decay, x1, epsilon, sigma=ere, absolute_sigma=True, p0=[4, -1/600, 1])
print("Parameters", params)
print("Error parameters", np.sqrt(np.diag(cov)))

# Try except RuntimeError 
L=np.zeros(len(peaks))
count=0
for i in range(len(peaks)):
    try:       
        x_min = bin_centers[peaks[i]] - 0.08* bin_centers[peaks[i]]
        x_max = bin_centers[peaks[i]] + 0.08 * bin_centers[peaks[i]]
        r = [x_min, x_max]
        hist1, bin1, bined1 = diff(e1, e2, 1, 1, 20, r)
        count=count+1
        popt = fit_diff(hist1, bin1, x_min, x_max, np.max(hist1))
        fit_range = (bin1 > x_min) & (bin1 < x_max)
        x_fit = np.linspace(np.min(bin1[fit_range]), np.max(bin1[fit_range]), 1000)
        y_fit = hist1[fit_range]
    except RuntimeError as e:
        if 'Optimal parameters not found' in str(e):
            continue
        else:
            raise e
        
#EXECUTE PROGRAM FOR FILE IN FOLDER
import os
def execute_program(file_path):
    print(f"Executing program for file: {file_path}")
    E=open_file(file_path, "0")
    plt.hist(E, range=[0, 8000], bins=1000, histtype='step', facecolor='g', alpha=0.75, label=file_path)
    print(len(E), file_path)
    #plt.show()
def run_for_all_files(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            execute_program(file_path)
folder_path = r"C:\Users\IL MIO PC\Desktop\DAVIDE\pythonmat\EXP 2\time"
run_for_all_files(folder_path)

# Concatanate
trgbegin=np.concatenate((trgbegin, tb[i]))

# Difference 
np.diff(trgbegin)

# Find peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from scipy.datasets import electrocardiogram

ecg = electrocardiogram()

Y=ecg

find_peaks(Y)
Y = Y + 10e-2 * np.random.randn(len(Y))
peaks, _ = find_peaks(Y, height=0, distance=100)
plt.plot(Y)
plt.plot(peaks, Y[peaks], ".")
plt.xlim(0, 5000)
plt.show()
print(Y[peaks])

# Plot with errorbars
plt.errorbar(h, j, e, color='red', fmt='.')

# Plot from parameters line
f=np.linspace(np.min(h), np.max(h), 1000)
plt.plot(f, exponential_decay(f, *params))

# Subplots
for i in range(6):
    plt.subplot(3, 2, i + 1)  
    plt.scatter(x_values, y_values[:, i], marker='x', s=6, label="near")
    plt.scatter(z_values, w_values[:, i], marker='x', s=6, label="far")
    plt.grid(True)
    if i>3:
      plt.xlabel(x_labels[0])  
    plt.ylabel(y_labels[i]) 
    plt.minorticks_on()
plt.show()

# Plot grid 
plt.grid(True)

# Gaussian 2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
x = 2*np.random.normal(size=500000)
y = np.random.normal(size=500000)

hist, xedges, yedges = np.histogram2d(x, y, bins=(100, 100), density=True)
def custom_2d_fit_function(coords, *params):
    x, y = coords
    return params[0] * np.exp(-(x/params[1])**2-(y/params[2])**2)

x_centers = 0.5 * (xedges[1:] + xedges[:-1])
y_centers = 0.5 * (yedges[1:] + yedges[:-1])
xdata, ydata = np.meshgrid(x_centers, y_centers)
zdata = hist.flatten()
area=100*100
print(type(x_centers))
initial_guess = [700, 1, 1]
params, cov = curve_fit(custom_2d_fit_function, (xdata.flatten(), ydata.flatten()), zdata*area, p0=initial_guess)
print(params)
print(np.sqrt(np.diag(cov)))

x_range = np.linspace(-5, 5, 200)
y_range = np.linspace(-2, 2, 200)
xx, yy = np.meshgrid(x_range, y_range)
zz = custom_2d_fit_function((xx, yy), *params)
plt.hist2d(x, y, bins=(100, 100), cmap=plt.cm.jet)
plt.colorbar()
#plt.contour(xx, yy, zz, colors='r', alpha=1)
import matplotlib.patches as patches
rectangle = patches.Rectangle((-1, -1), 2, 2, edgecolor='blue', facecolor='none', linewidth=2)
plt.gca().add_patch(rectangle)
X, Y = np.meshgrid(x_range, y_range)
Z = custom_2d_fit_function((X, Y), *params)
def fmt(x):
    s = f"{x:.0f}"
    return rf"{s} " if plt.rcParams["text.usetex"] else f"{s}"
CS = plt.contour(X, Y, Z, colors='black')
plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
plt.xlim(-4, 4)
plt.ylim(-2, 2)
plt.show()

# Difference histograms
def diff(e1, e2, time1, time2, bop, r):
    a=np.full(len(e1), 6)
    #b=np.full(len(e2), 1/time2)
    hist1, bin_edges = np.histogram(e1, bins=bop, weights=a, range=r)
    hist2, _ = np.histogram(e2, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_diff = hist1 - hist2
    hist_diff = np.maximum(hist_diff, 0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return hist_diff, bin_centers, bin_edges

# Fit difference histograms 
def fit_diff(hist_diff, bin_centers, x_min, x_max, K):
    p0 = [100, (x_max+x_min)/2, 20, 0, 0]# Initial parameter values
    fit_range = (bin_centers> x_min) & (bin_centers < x_max)
    x_fit = bin_centers[fit_range]
    y_fit = hist_diff[fit_range]
    popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0)
    #print("Fit parameters and errors:")
    perr = np.sqrt(np.diag(pcov))
    """for param, error in zip(popt, perr):
       print(f"Parameter: {param} Error: {error}")"""
    k=(math.sqrt(abs(2*math.pi))*popt[0]*popt[2])/((x_max-x_min)/len(fit_range))
    print(popt[1], perr[1],  abs(k), math.sqrt((k/popt[0]*perr[0])**2+(k/popt[2]*perr[2])**2))
    return popt

# Find peaks 
r=[1, 800]
from scipy.signal import find_peaks
hist_diff, bin_centers, bin_edges=diff(e1, e2, 1, 6, 200, r)
Y=hist_diff
find_peaks(Y)
peaks, _ = find_peaks(Y, height=1, distance=1, prominence=300)
plt.plot(bin_centers[peaks], Y[peaks], ".", color='orange')
plt.bar(bin_centers, hist_diff, width=np.diff(bin_edges), align='center', alpha=0.75)
#plt.yscale('log')
plt.xlabel("Energy [keV]")
plt.ylabel("Events")
plt.minorticks_on()
plt.show()
print(np.sum(hist_diff), math.sqrt(np.sum(hist_diff)))

#CREATE NUCLEAR LEVEL SCHEMES
plt.figure(figsize=(12, 6))
# Plotting Scheme 1
plt.subplot(1, 2, 1)
plt.plot([0, 1], [0, max_energy], 'w.')  # Creating a blank plot for setting the scale
for i, (energy, spin) in enumerate(zip(energies_scheme_1, spins_scheme_1)):
    plt.plot([0.1, 0.9], [energy, energy], 'k-', lw=2)
    plt.text(1.05, energy, f'{energy} keV, I=' f'{spin}+', ha='left', va='center', fontsize=20)
plt.title('NUCLEAR DATA', fontsize=20)
plt.ylim(-100, max_energy + 100)
plt.axis('off')
# Plotting Scheme 2
plt.subplot(1, 2, 2)
plt.plot([0, 1], [0, max_energy], 'w.')  # Creating a blank plot for setting the scale
for i, (energy, spin) in enumerate(zip(energies_scheme_2, spins_scheme_2)):
    plt.plot([0.1, 0.9], [energy, energy], 'k-', lw=2)
    plt.text(1.05, energy, f'{energy} keV  I=' f'{spin}+', ha='left', va='center', fontsize=20)
plt.title('KSHELL', fontsize=20)
plt.ylim(-100, max_energy + 100)
plt.axis('off')
plt.tight_layout()
plt.show()

# Bar plot
cha=np.arange(0, np.max(multiplicity))
for event in range(len(tb)):
    for channel in range(len(tb[0])):
        if tb[event][channel]>0:
            cha[channel]=cha[channel]+1
indexes1 = np.arange(0, np.max(multiplicity))
plt.bar(indexes1+1, cha, width=1)
plt.title('Plot Events over Channel Muons')
plt.minorticks_on()
plt.xlabel('Channel')
plt.ylabel('Events')

plt.show()

# Try images
import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
x = np.random.normal(size=500)
y = x * 3 + np.random.normal(size=500)
z = np.random.normal(size=500)+y/2
ax= plt.axes(projection="3d")
ax.scatter(x, y, z)
ax.set_title("3D PLOT")
plt.show()

plt.figure(2)
ax= plt.axes(projection="3d")
x=np.arange(-5, 5, 0.1)
y=np.arange(-10, 0, 0.1)
X, Y=np.meshgrid(x, y)
Z=np.sin(X)*np.cos(Y)
ax.plot_surface(X, Y, Z, cmap="Spectral")
plt.show()
plt.savefig("imagetry.png")

plt.figure(3)
import random
heads_tails=[0, 0]
for _ in range(100):
    heads_tails[random.randint(0, 1)]+=1
    plt.bar(["Heads", "Tails"], heads_tails, color=["red", "blue"])
    plt.pause(0.000001)
plt.show()

plt.figure(4)
from mpl_toolkits.mplot3d import axes3d
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.1)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
for angle in range(0, 360):
   ax.view_init(30, angle)
   plt.draw()
   plt.pause(.001)
plt.show()

# Vertical lines
plt.axvline(x_min, color = 'y', linewidth=0.5, label = 'axvline - full height')
plt.axvline(x_max, color = 'y', linewidth=0.5, label = 'axvline - full height')

# 2D fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
x = 2*np.random.normal(size=500000)
y = x+np.random.normal(size=500000)

hist, xedges, yedges = np.histogram2d(x, y, bins=(100, 100), density=True)
def custom_2d_fit_function(coords, *params):
    x, y = coords
    return params[0] * np.exp(params[1]*x**2+params[2]*y**2+params[3]*x*y)

x_centers = 0.5 * (xedges[1:] + xedges[:-1])
y_centers = 0.5 * (yedges[1:] + yedges[:-1])
xdata, ydata = np.meshgrid(x_centers, y_centers)
zdata = hist.flatten()
area=100*100
initial_guess = [700, -1, -1, -1]
params, cov = curve_fit(custom_2d_fit_function, (xdata.flatten(), ydata.flatten()), zdata*area, p0=initial_guess)
print(params)
print(np.sqrt(np.diag(cov)))
x_range = np.linspace(-5, 5, 200)
y_range = np.linspace(-2, 2, 200)
xx, yy = np.meshgrid(x_range, y_range)
zz = custom_2d_fit_function((xx, yy), *params)
plt.hist2d(x, y, bins=(100, 100), cmap=plt.cm.jet)
plt.colorbar()
#plt.contour(xx, yy, zz, colors='r', alpha=1)
import matplotlib.patches as patches
rectangle = patches.Rectangle((-1, -1), 2, 2, edgecolor='blue', facecolor='none', linewidth=2)
plt.gca().add_patch(rectangle)
X, Y = np.meshgrid(x_range, y_range)
Z = custom_2d_fit_function((X, Y), *params)
def fmt(x):
    s = f"{x:.0f}"
    return rf"{s} " if plt.rcParams["text.usetex"] else f"{s}\n"
contour_levels = np.arange(0, 700, 100)
CS = plt.contour(X, Y, Z, colors='black')
plt.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
plt.xlim(-4, 4)
plt.ylim(-2, 2)
plt.show()

# Bar plot 3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the custom 2D fit function
def custom_2d_fit_function(coords, *params):
    x, y = coords
    return params[0] * np.exp(-(x / params[1])**2 - (y / params[2])**2)

# Generate a meshgrid for x and y
x_range = np.linspace(-5, 5, 50)
y_range = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x_range, y_range)
coords = (X, Y)

# Set example parameters (replace with your fit results)
example_params = params

# Evaluate the function with the example parameters
Z = custom_2d_fit_function(coords, *example_params)

# Plot the 3D bar plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

xpos, ypos = X.flatten(), Y.flatten()
zpos = np.zeros_like(xpos)
dx, dy = 0.5, 0.5
dz = Z.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

ax.set_title('Custom 2D Fit Function (Bar Plot)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Function Value')

plt.show()

# 3D with MeshGrid
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xbin=50
ybin=50
hist, xedges, yedges = np.histogram2d(x, y, bins=(xbin, ybin))

xpos, ypos = np.meshgrid(xedges[:-1] + abs(np.max(x)-np.min(x))/xbin, yedges[:-1] + abs(np.max(y)-np.min(y))/ybin, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
print(abs(np.max(x)-np.min(x))/xbin)
# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()

# 3D surface MeshGrid
ax= plt.axes(projection="3d")
x=np.arange(-5, 5, 0.1)
y=np.arange(-10, 0, 0.1)
X, Y=np.meshgrid(x, y)
Z=np.sin(X)*np.cos(Y)
ax.plot_surface(X, Y, Z, cmap="Spectral")
plt.show()

# Open leaf 
def open_file(filename, leaf):
    with uproot.open(filename) as file:
        event = file["tree"]
        return event[leaf].array(library="np")
    
tree.show()
active_ch= tree['active_ch'].array(library='np')