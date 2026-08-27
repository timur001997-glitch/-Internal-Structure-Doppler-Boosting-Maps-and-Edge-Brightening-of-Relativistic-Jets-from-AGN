import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from numba import jit

start_time = time.time()  # start time of calculations

sigma_M = 100 #Michel magnetization parameter
Gamma_in = 2 #hydrodynamic Lorentz factor on the jet axis
h_0 = 1*10**(1) #magnetic field at the distance R_L from the central engine in G
c = 3*10**10 #the speed of light in cm/s
G = 6.67*10**(-8) #the gravitational constant in the CGS system
M = 1*10**5 * 2*10**33 #the mass of the compact object in g
#M = 2*10**33 #the mass of the Sun in g
r_g = 2*G*M/c**2 #gravitational radius in cm
R_L = 10*r_g #the radius of the light cylinder in cm
R_L = 10**(9)
m_e = 9.1*10**(-28) #the mass of an electron in g
q = 4.8 * 10**(-10) #the electron charge in the CGS system
t_0 = R_L/c #dimensionless time

P_0 = 4.5 * 10**(-10) #pressure in dyn/cm^2 for M87
z_00 = 220 * 3.086 * 10**18 #the characteristic distance along the jet axis in cm for M87
z_00 = z_00/10**16 #the characteristic dimensionless distance along the jet axis in cm for M87
Psi_0 = np.pi #full magnetic flux!
r_jet_cr = np.sqrt(Gamma_in*sigma_M) #in light cylinders R_L (dimensionless)
r_core = 5 #in light cylinders R_L (dimensionless)

delta = 10**(-4) #anti-breakage program - small delta supplement

Omega_0 = 1

print(r_jet_cr)

#The direction of the line of sight in the laboratory reference frame
Theta = np.pi*17/180 #the angle between the z axis of the jet rotation and the line of sight
n_x = np.sin(Theta)
n_y = 0
n_z = np.cos(Theta)

#Formulas for EM fields through magnetic flux
@jit(nopython=True)
def norm(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return r

@jit(nopython=True)
def r_jet(z):
    r_jet = (Psi_0**2*h_0**2/(8*np.pi**3*P_0*z_00**2))**(1/4) * z**(2/4)
#    Theta_jet = 0.2
#    r_jet = Theta_jet * z
    return r_jet #the shape of the jet in R_L units (dimensionless)

@jit(nopython=True)
def k(z):
    k = 4.5*sigma_M*(r_jet(z)/(6*sigma_M))**(0.65) * (1 + (r_jet(z)/(6*sigma_M))**2)**(0.15)
    return k

@jit(nopython=True)
def alpha(z):
    a = 0.52*sigma_M**(0.18)
    c_0 = 2.8 * sigma_M**(0.22)
    if (r_jet_cr - r_jet(z)) >= 0:
        alpha = 0
    else:
        alpha = 1.3 * (np.log(r_jet(z) - r_jet_cr + 1)/c_0)**a * (1 + (np.log(r_jet(z) - r_jet_cr + 1)/c_0)**6)**((0.33 - a)/6)
    return alpha

@jit(nopython=True)
def B_0(z):
    B_0 = Psi_0/(np.pi*r_core**2) * (1 - alpha(z)/2)/((1 + r_jet(z)**2/r_core**2)**(1-alpha(z)/2) - 1)
    return B_0

@jit(nopython=True)
def Psi(x,y,z):
    rho = norm(x,y,0)
    if (r_jet_cr - r_jet(z)) >= 0:
        Psi = Psi_0 * (rho/r_jet(z))**2
    else:
        Psi = np.pi*B_0(z)*r_core**2/(1-alpha(z)/2) * ((1 + rho**2/r_core**2)**(1 - alpha(z)/2) - 1)
    return Psi

@jit(nopython=True)
def Partial_Derivative(func, x,y,z, var_index, h=delta):  
    if var_index == 0:
        f_plus = func(x + h, y, z)
        f_minus = func(x - h, y, z)
    elif var_index == 1:
        f_plus = func(x, y + h, z)
        f_minus = func(x, y - h, z)
    else:
        f_plus = func(x, y, z + h)
        f_minus = func(x, y, z - h)  
    return (f_plus - f_minus) / (2 * h)

@jit(nopython=True)
def Omega(x,y,z):
    Omega = Omega_0 * np.sqrt(1 - Psi(x, y, z)/Psi_0)
    return Omega

#The maximum Lorentz factor
@jit(nopython=True)
def Gamma_max(x,y,z):
    Gamma_max = Gamma_in + 2*sigma_M*Psi(x, y, z)/Psi_0 * (1 - Psi(x, y, z)/Psi_0)
    return Gamma_max

@jit(nopython=True)
def epsilon(x,y,z):
    epsilon = 1/2/Gamma_max(x,y,z)**2
#    epsilon = 1/sigma_M**2
#    epsilon = 0
    return epsilon

@jit(nopython=True)
def E_x(x,y,z):
    E_x = -Omega(x,y,z)/(2*np.pi) * Partial_Derivative(Psi,x,y,z,0)
    return E_x

@jit(nopython=True)
def E_y(x,y,z):
    E_y = -Omega(x,y,z)/(2*np.pi) * Partial_Derivative(Psi,x,y,z,1)
    return E_y

@jit(nopython=True)
def E_z(x,y,z):
    E_z = -Omega(x,y,z)/(2*np.pi) * Partial_Derivative(Psi,x,y,z,2)
    return E_z

@jit(nopython=True)
def B_p_x(x,y,z):
    rho = norm(x,y,0)
    B_p_x = -1/(2*np.pi*rho) * x/rho * Partial_Derivative(Psi,x,y,z,2)
    return B_p_x

@jit(nopython=True)
def B_phi_x(x,y,z):
    rho = norm(x,y,0)
    B_phi_x = (1+epsilon(x,y,z))*Omega(x,y,z)/(2*np.pi) * y/rho * norm(Partial_Derivative(Psi,x,y,z,0), Partial_Derivative(Psi,x,y,z,1), Partial_Derivative(Psi,x,y,z,2))
    return B_phi_x

@jit(nopython=True)
def B_x(x,y,z):
    return B_p_x(x,y,z) + B_phi_x(x,y,z)

@jit(nopython=True)
def B_p_y(x,y,z):
    rho = norm(x,y,0)
    B_p_y = -1/(2*np.pi*rho) * y/rho * Partial_Derivative(Psi,x,y,z,2)
    return B_p_y

@jit(nopython=True)
def B_phi_y(x,y,z):
    rho = norm(x,y,0)
    B_phi_y = -(1+epsilon(x,y,z))*Omega(x,y,z)/(2*np.pi) * x/rho * norm(Partial_Derivative(Psi,x,y,z,0), Partial_Derivative(Psi,x,y,z,1), Partial_Derivative(Psi,x,y,z,2))
    return B_phi_y

@jit(nopython=True)
def B_y(x,y,z):
    return B_p_y(x,y,z) + B_phi_y(x,y,z)

@jit(nopython=True)
def B_p_z(x,y,z):
    rho = norm(x,y,0)
    B_p_z = 1/(2*np.pi*rho) * (x/rho*Partial_Derivative(Psi,x,y,z,0)+y/rho*Partial_Derivative(Psi,x,y,z,1))
    return B_p_z

@jit(nopython=True)
def B_z(x,y,z):
    return B_p_z(x,y,z)

#Formulas for the components of the dimensionless hydrodynamic (drift) velocity
@jit(nopython=True)
def V_x(x,y,z):
    B = norm(B_x(x, y, z),B_y(x, y, z),B_z(x, y, z))
    V_x = (E_y(x, y, z)*B_z(x, y, z) - E_z(x, y, z)*B_y(x, y, z))/B**2
    return V_x

@jit(nopython=True)
def V_y(x,y,z):
    B = norm(B_x(x, y, z),B_y(x, y, z),B_z(x, y, z))
    V_y = (E_z(x, y, z)*B_x(x, y, z) - E_x(x, y, z)*B_z(x, y, z))/B**2
    return V_y

@jit(nopython=True)
def V_z(x,y,z):
    B = norm(B_x(x, y, z),B_y(x, y, z),B_z(x, y, z))
    V_z = (E_x(x, y, z)*B_y(x, y, z) - E_y(x, y, z)*B_x(x, y, z))/B**2
    return V_z

#Hydrodynamic Lorentz factor
@jit(nopython=True)
def Gamma(x,y,z):
    V = norm(V_x(x, y, z),V_y(x, y, z),V_z(x, y, z))
    Gamma = 1/np.sqrt(1 - V**2)
    Gamma = np.sqrt(Gamma_in**2 - 1 + Gamma**2)
    return Gamma

#The Doppler factor
@jit(nopython=True)
def D(x,y,z): 
    D = 1/Gamma(x, y, z)/(1 - V_x(x, y, z)*n_x - V_y(x, y, z)*n_y - V_z(x, y, z)*n_z)
    return D

#The local magnetization
@jit(nopython=True)
def sigma(x,y,z):
    rho = norm(x,y,0)
    sigma = (1 - rho**2/r_jet(z)**2) * rho**2/r_jet(z)**2 * sigma_M/Gamma(x, y, z)
    return sigma

z_min = 1
print(r_jet(z_min))
z_max = z_min + 3000
z_0 = 100
x_0 = 0
y_0 = 0
#Construction of the uniform square grid NxN, inside which lies the jet
N = 100
Z = np.linspace(z_min, z_max,N)
Y = np.linspace(-r_jet(z_max), r_jet(z_max),N)
D_data = []
S_data = []
G_data = []
for i in range(N):
    z_0 = Z[i]
    for j in range(N):
        y_0 = Y[j]
        if np.sqrt(x_0**2 + y_0**2)<=r_jet(z_0):
            D_data = np.append(D_data,D(x_0, y_0, z_0))
            S_data = np.append(S_data,sigma(x_0, y_0, z_0))
            G_data = np.append(G_data,Gamma(x_0, y_0, z_0))
        else:
            D_data = np.append(D_data,0)
            S_data = np.append(S_data,0)
            G_data = np.append(G_data,1)

print(max(D_data))
print(max(S_data))
print(max(G_data))
D_data = np.reshape(D_data, (N, N))
D_data = np.transpose(D_data)
S_data = np.reshape(S_data, (N, N))
S_data = np.transpose(S_data)
G_data = np.reshape(G_data, (N, N))
G_data = np.transpose(G_data)
Y = np.flip(Y)
#T_data = np.flip(T_data)

z_min, z_max = min(Z), max(Z)
y_min, y_max = min(Y), max(Y)
D_min, D_max = D_data.min(),D_data.max()
S_min, S_max = S_data.min(),S_data.max()
G_min, G_max = G_data.min(),G_data.max()

#Building the color scale for the Doppler factor
colors = [
    (0.15, 0.25, 0.45),
    (0.25, 0.4, 0.6),
    (0.65, 0.45, 0.3),
    (0.85, 0.55, 0.25),
    (0.95, 0.7, 0.3),
    (1.0, 0.92, 0.65)
]

color_positions = [0, 0.15, 0.3, 0.5, 0.8, 1.0]

final_cmap = LinearSegmentedColormap.from_list(
    'blue_to_orange_gold_light_peak', 
    list(zip(color_positions, colors)), 
    N=256
)

#The Doppler factor map
plt.figure(figsize=(8, 4.5))
im = plt.imshow(D_data,origin='lower',cmap=final_cmap,extent=[z_min, z_max, y_min, y_max],aspect="auto")
#Adding text
if z_max >= 300000:
    plt.text(-62500, -1375, 'c)', fontsize=26, color='black')
elif z_max >= 3000:
    plt.text(-200, -80, 'c)', fontsize=26, color='black')
#    plt.text(-2750, -290, 'c)', fontsize=26, color='black')
#    plt.text(-2000, -255, 'c)', fontsize=26, color='black')
else:
    plt.text(-20, -26, 'd)', fontsize=26, color='black')
#Adding the color scale (colorbar)
cbar = plt.colorbar(im)
cbar.ax.set_title('$D$', fontsize=15)
cbar.ax.tick_params(which='major',labelsize=14)
plt.xlabel('$z/R_{L}$', fontsize=15)
plt.ylabel('$y/R_{L}$', fontsize=15)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(5))  # maximum of 5 divisions
plt.title('$\sigma_{M}=100, \Gamma_{in}=2, z_{st}/R_{L}=1, \\Theta = {17}^{\circ}$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

#The magnetization map
plt.figure(figsize=(8, 4.5))
im = plt.imshow(S_data,origin='lower',cmap='jet',extent=[z_min, z_max, y_min, y_max],aspect="auto")
#Adding text
if z_max >= 300000:
    plt.text(-62500, -1375, 'a)', fontsize=26, color='black')
elif z_max >= 3000:
    plt.text(-200, -80, 'a)', fontsize=26, color='black')
#   plt.text(-2750, -290, 'a)', fontsize=26, color='black')
#Adding the color scale (colorbar)
cbar = plt.colorbar(im)
cbar.ax.set_title('$\sigma$', fontsize=15)
cbar.ax.tick_params(which='major',labelsize=14)
plt.xlabel('$z/R_{L}$', fontsize=15)
plt.ylabel('$y/R_{L}$', fontsize=15)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(5))  # maximum of 5 divisions
plt.title('$\sigma_{M}=100,\Gamma_{in}=2, z_{st}/R_{L}=1$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

#The Lorentz factor map
plt.figure(figsize=(8, 4.5))
im = plt.imshow(G_data,origin='lower',cmap='viridis',extent=[z_min, z_max, y_min, y_max],aspect="auto")
#Adding text
if z_max >= 300000:
    plt.text(-62500, -1375, 'b)', fontsize=26, color='black')
elif z_max >= 3000:
    plt.text(-200, -80, 'b)', fontsize=26, color='black')
#    plt.text(-2750, -290, 'b)', fontsize=26, color='black')
#Adding the color scale (colorbar)
cbar = plt.colorbar(im)
cbar.ax.set_title('$\Gamma$', fontsize=15)
cbar.ax.tick_params(which='major',labelsize=14)
plt.xlabel('$z/R_{L}$', fontsize=15)
plt.ylabel('$y/R_{L}$', fontsize=15)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(5))  # maximum of 5 divisions
plt.title('$\sigma_{M}=100,\Gamma_{in}=2, z_{st}/R_{L}=1$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

end_time = time.time()  # the end time of the calculation
execution_time = end_time - start_time  # calculating the program runtime
 
print(f"Program execution time: {execution_time} seconds")
