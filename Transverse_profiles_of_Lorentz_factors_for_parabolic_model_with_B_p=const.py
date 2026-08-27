import numpy as np
import matplotlib.pyplot as plt
import time
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
r_core = 1 #in light cylinders R_L (dimensionless)

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
    return 1.3 * (np.log(r_jet(z) - r_jet_cr + 1)/c_0)**a * (1 + (np.log(r_jet(z) - r_jet_cr + 1)/c_0)**6)**((0.33 - a)/6)

@jit(nopython=True)
def B_0(z):
    B_0 = Psi_0/(np.pi*r_core**2) * (1 - alpha(z)/2)/((1 + r_jet(z)**2/r_core**2)**(1-alpha(z)/2) - 1)
    return B_0

@jit(nopython=True)
def Psi(x,y,z):
    rho = norm(x,y,0)
    Psi = Psi_0 * (rho/r_jet(z))**2
#    Psi = np.pi*B_0(z)*r_core**2/(1-alpha(z)/2) * ((1 + rho**2/r_core**2)**(1 - alpha(z)/2) - 1)
    return Psi

#def Partial_Derivative(func, input_values, var_index, h=delta):
#    original_val = input_values[var_index]
#    input_values[var_index] = original_val + h
#    f_plus_h = func(*input_values)
#    input_values[var_index] = original_val - h
#    f_minus_h = func(*input_values)
#    input_values[var_index] = original_val
#    return (f_plus_h - f_minus_h) / (2 * h)

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

#Hydrodynamic Lorentz factor in the case of Omega=const
def Gamma_dr_linear(x_r):
    Gamma_dr = np.sqrt(1 + x_r**2)
    Gamma_dr = np.sqrt(Gamma_in**2 - 1 + Gamma_dr**2)
    return Gamma_dr

z_0 = 100
x_0 = 0
y_0 = 0
#Construction of the uniform one-dimensional grid N, inside which lies the jet
N = 10000
Y = np.linspace(-r_jet(z_0), r_jet(z_0),N)
D_data = []
S_data = []
G_data = []
G_lin_data = []
for j in range(N):
    y_0 = Y[j]
    if np.sqrt(x_0**2 + y_0**2)<=r_jet(z_0):
        D_data = np.append(D_data,D(x_0, y_0, z_0))
        S_data = np.append(S_data,sigma(x_0, y_0, z_0))
        G_data = np.append(G_data,Gamma(x_0, y_0, z_0))
        G_lin_data = np.append(G_lin_data,Gamma_dr_linear(y_0))
    else:
        D_data = np.append(D_data,0)
        S_data = np.append(S_data,0)
        G_data = np.append(G_data,1)

print(max(D_data))
print(max(S_data))
print(max(G_data))

#Profile of the Lorentz factor and the Doppler factor
fig, ax = plt.subplots(dpi = 100)
plt.plot(Y,G_data,color = 'blue', linewidth = 2, linestyle = '-',label='$\Gamma$') 
plt.plot(Y,G_lin_data,color = 'red', linewidth = 2, linestyle = '--',label='$\Gamma ≈ x$')
plt.plot(Y,D_data,color = 'green', linewidth = 2, linestyle = '-',label='$D$') 
plt.xlabel('$x$', fontsize=15)
plt.ylabel('$\Gamma$', fontsize=15)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(5))  # maximum of 5 divisions
ax.yaxis.set_major_locator(MaxNLocator(5))  # maximum of 5 divisions
plt.title('$\sigma_{M}=100, \Gamma_{in}=2, \\Theta = {17}^{\circ}, z = 10^{2}R_{L}$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.show()

end_time = time.time()  # the end time of the calculation
execution_time = end_time - start_time  # calculating the program runtime
 
print(f"Program execution time: {execution_time} seconds")
