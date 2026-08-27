import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

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

#epsilon = 1/sigma_M**2
Omega_0 = 1

print(r_jet_cr)

#The system of two ordinary differential equations for M2 and Y, where
#M2 - the square of the Alfven Mach number, Y - dimensionless magnetic flux
#The system for new integrals of motion Omega(Psi) and L(Psi)
def f(X, x_r):
   Y = X[0]
   M2 = X[1]
   A = 1 - x_r**2*(1 - Y/sigma_M) - M2
   diff_1 = np.sqrt(1/A**2 * (Gamma_in**2*x_r**2*(A - M2)/M2**2 + x_r**2*(Gamma_in + 2*Y*(1 - Y/sigma_M))**2 - 4*Y**2*(1 - Y/sigma_M) - x_r**2*A**2/M2**2))
   diff_2 = 1/(Gamma_in**2 + x_r**2*(1 - Y/sigma_M) - 1) * (2*x_r*M2*(1 - Y/sigma_M) - Gamma_in**2*x_r*M2*(1 - Y/sigma_M)/A + 4*Y**2*M2**3*(1 - Y/sigma_M)/x_r**3/A - 1/sigma_M*x_r**2*M2*diff_1)
   return [diff_1, diff_2]

#The Lorentz factor from the Grad-Shafranov equation
def Gamma_GS(x_r, M2, Y):
    Gamma_GS = (Gamma_in - M2*(Gamma_in + 2*Y*(1-Y/sigma_M)))/(1 - x_r**2*(1-Y/sigma_M) - M2)
    return Gamma_GS

#The maximum Lorentz factor
def Gamma_max(Y):
    Gamma_max = Gamma_in + 2*Y * (1 - Y/sigma_M)
    return Gamma_max

def epsilon(Y):
    epsilon_func = 1/2/Gamma_max(Y)**2
    return epsilon_func

#Hydrodynamic Lorentz factor in the framework of MHD
def Gamma_dr_MHD(B_p,B_phi,E):
    B = np.sqrt(B_p**2 + B_phi**2)
    beta_dr = E/B
    Gamma_dr = 1/np.sqrt(1 - beta_dr**2)
    Gamma_dr = np.sqrt(Gamma_in**2 - 1 + Gamma_dr**2)
    return Gamma_dr

#Hydrodynamic Lorentz factor taking into account epsilon
def Gamma_dr_epsilon(x_r, Y):
    beta_dr = x_r*np.sqrt(1 - Y/sigma_M)/np.sqrt(1 + (1+epsilon(Y))**2 * x_r**2*(1 - Y/sigma_M))
    Gamma_dr = 1/np.sqrt(1 - beta_dr**2)
    Gamma_dr = np.sqrt(Gamma_in**2 - 1 + Gamma_dr**2)
    return Gamma_dr

#Hydrodynamic Lorentz factor in the case of Omega=const
def Gamma_dr_linear(x_r):
    Gamma_dr = np.sqrt(1 + x_r**2)
    Gamma_dr = np.sqrt(Gamma_in**2 - 1 + Gamma_dr**2)
    return Gamma_dr

def I(x_r, Y, M2):
    I = Omega_0 * Psi_0/4/np.pi * 1/sigma_M * np.sqrt(1 - Y/sigma_M) * (2*Y - x_r**2*(Gamma_in + 2*Y*(1 - Y/sigma_M)))/(1 - x_r**2*(1 - Y/sigma_M) - M2)
    return I

def Psi(Y):
    Psi = Psi_0/sigma_M * Y
    return Psi

def Omega_F(Y):
    Omega_F = Omega_0 * np.sqrt(1 - Y/sigma_M)
    return Omega_F

#Building the local magnetization profile
def L(Y):
    L = Omega_0 * Psi(Y)/(4*np.pi**2) * np.sqrt(1 - Psi(Y)/Psi_0)
    return L

def E_Bern(x_r, Y, M2):
    E = Omega_F(Y)*I(x_r, Y, M2)/(2*np.pi) + 1/sigma_M * Omega_0**2*Psi_0/(8*np.pi**2) * Gamma_GS(x_r, M2, Y)
    return E

def sigma(x_r,Y,M2):
    sigma = Omega_F(Y) * (L(Y) - Omega_F(Y)*x_r**2*E_Bern(x_r, Y, M2)) / (E_Bern(x_r, Y, M2) - Omega_F(Y)*L(Y) - M2*E_Bern(x_r, Y, M2))
    return sigma


N = 100000 #The number of spatial points for determining the Lorentz factor

#Introduction of the initial conditions
rho_0 = 0.001
#M2_0 = 3
M2_0 = 3
Y_0 = 1/2*Gamma_in/M2_0*rho_0**2
ics=[Y_0, M2_0]
#rho = np.linspace(rho_0,53.9,N) #distance from the jet axis in R_L
rho = np.linspace(rho_0,53.9,N) #distance from the jet axis in R_L

solution = odeint(f, ics, rho)
Y = solution[:,0]
M2 = solution[:,1]
print(Y)
print(M2)

print(epsilon(Y))

#Calculate the values of electromagnetic fields
Psii = Psi(Y)
Omega_FF = Omega_F(Y)
II = I(rho, Y, M2)
K = len(Y)
rho_MHD = np.zeros(K-1)
#Configuration of electromagnetic fields through vectors
B_p = np.zeros(K-1)
E = np.zeros(K-1)
B_phi = np.zeros(K-1)
for j in range(K):
    if j>=1:
        rho_MHD[j-1] = rho[j]
        delta_rho = rho[j] - rho[j-1]
        delta_Psi = Psii[j] - Psii[j-1]
        grad_Psi = delta_Psi/delta_rho
        B_p[j-1] = grad_Psi/2/np.pi/rho[j]
        E[j-1] = Omega_FF[j] * grad_Psi/2/np.pi
        B_phi[j-1] = 2*II[j]/rho[j]

sigma_print = sigma(rho, Y, M2)

fig, ax = plt.subplots(dpi = 100)
plt.plot(rho,sigma_print,color = 'black', linewidth = 2, linestyle = '-') 
plt.xlabel(r'$\varpi / R_{L}$', fontsize=15)
plt.ylabel('$\sigma$', fontsize=15)
plt.title('$\sigma_{M}=100,\Gamma_{in}=2, M^{2}_{0}=15$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()

fig, ax = plt.subplots(dpi = 100)
plt.plot(rho,Y,color = 'black', linewidth = 2, linestyle = '-') 
plt.xlabel(r'$\varpi / R_{L}$', fontsize=15)
plt.ylabel('$Y$', fontsize=15)
plt.title('$\sigma_{M}=100,\Gamma_{in}=2, M^{2}_{0}=3$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()

fig, ax = plt.subplots(dpi = 100)
plt.plot(rho,M2,color = 'blue', linewidth = 2, linestyle = '-') 
plt.xlabel(r'$\varpi / R_{L}$', fontsize=15)
plt.ylabel('$M^{2}$', fontsize=15)
plt.title('$\sigma_{M}=100,\Gamma_{in}=2, M^{2}_{0}=3$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()

Gamma_dr_MHD_print = Gamma_dr_MHD(B_p, B_phi, E)
Gamma_GS_print = Gamma_GS(rho,M2,Y)
Gamma_max_print = Gamma_max(Y)
Gamma_dr_epsilon_print = Gamma_dr_epsilon(rho,Y)
Gamma_dr_linear_print = Gamma_dr_linear(rho)
print(Gamma_GS_print)
print(Gamma_dr_MHD_print)
print(Gamma_dr_epsilon_print)

fig, ax = plt.subplots(dpi = 100)
plt.plot(rho,Gamma_GS_print,color = 'red', linewidth = 2, linestyle = '-',label='$\Gamma$') 
plt.plot(rho_MHD,Gamma_dr_MHD_print,color = 'blue', linewidth = 2, linestyle = '-',label='$\Gamma_{dr, MHD}$') 
plt.plot(rho,Gamma_dr_epsilon_print,color = 'green', linewidth = 2, linestyle = '-.',label='$\Gamma_{dr, \epsilon}$')
#plt.plot(rho,Gamma_dr_linear_print,color = 'red', linewidth = 2, linestyle = '--',label='$\Gamma$ = x')  
plt.plot(rho,Gamma_max_print,color = 'black', linewidth = 2, linestyle = 'dotted',label='$\Gamma_{max}$') 
plt.xlabel('$x$', fontsize=15)
plt.ylabel('$\Gamma$', fontsize=15)
plt.title('$\sigma_{M}=100,\Gamma_{in}=2, M^{2}_{0}=3$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.show()

end_time = time.time()  # the end time of the calculation
execution_time = end_time - start_time  # calculating the program runtime.

print(f"Program execution time: {execution_time} seconds")
