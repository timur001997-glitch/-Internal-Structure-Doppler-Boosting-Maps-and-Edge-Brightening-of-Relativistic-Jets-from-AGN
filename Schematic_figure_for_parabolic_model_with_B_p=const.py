import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

start_time = time.time()  # время начала выполнения

sigma_M = 100
Gamma_in = 2 #гидродинамический лоренц-фактор на оси джета
h_0 = 1*10**(1) #магнитное поле на расстоянии одного R_L от центральной машины в Гс
c = 3*10**10 #скорость света в см/c
G = 6.67*10**(-8) #гравитационная постоянная в системе СГС
M = 1*10**5 * 2*10**33 #масса компактного объекта в г
#M = 2*10**33 #масса Солнца в г
r_g = 2*G*M/c**2 #гравитационный радиус в см
R_L = 10*r_g #радиус светового цилиндра в см
R_L = 10**(9)
m_e = 9.1*10**(-28) #масса электрона в г
q = 4.8 * 10**(-10) #заряд электрона в системе СГС
t_0 = R_L/c #обезразмеривание времени

P_0 = 4.5 * 10**(-10) #давление в дин/см^2 для M87
z_00 = 220 * 3.086 * 10**18 #характерное расстояние вдоль оси джета в см для M87
z_00 = z_00/10**16 #характерное безразмерное расстояние вдоль оси джета для M87
Psi_0 = np.pi #полный магнитный поток!
r_jet_cr = np.sqrt(Gamma_in*sigma_M) #в световых цилиндрах R_L (безразмерная)
r_core = 1 #в световых цилиндрах R_L (безразмерная)

delta = 10**(-2) #анти-поломка программы - малая добавка delta

Omega_0 = 1

print(r_jet_cr)

#Направление луча зрения в лабораторной СО
Theta = np.pi*17/180 #угол между осью z вращения джета и лучом зрения
n_x = np.sin(Theta)
n_y = 0
n_z = np.cos(Theta)

#Формулы для ЭМ полей через магнитный поток
def norm(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return r

def r_jet(z):
    r_jet = (Psi_0**2*h_0**2/(8*np.pi**3*P_0*z_00**2))**(1/4) * z**(2/4)
#    Theta_jet = 0.2
#    r_jet = Theta_jet * z
    return 1.*r_jet #форма джета в единицах R_L (безразмерная)

def k(z):
    k = 4.5*sigma_M*(r_jet(z)/(6*sigma_M))**(0.65) * (1 + (r_jet(z)/(6*sigma_M))**2)**(0.15)
    return k

def alpha(z):
    a = 0.52*sigma_M**(0.18)
    c_0 = 2.8 * sigma_M**(0.22)
    return 1.3 * (np.log(r_jet(z) - r_jet_cr + 1)/c_0)**a * (1 + (np.log(r_jet(z) - r_jet_cr + 1)/c_0)**6)**((0.33 - a)/6)

def B_0(z):
    B_0 = Psi_0/(np.pi*r_core**2) * (1 - alpha(z)/2)/((1 + r_jet(z)**2/r_core**2)**(1-alpha(z)/2) - 1)
    return B_0

def Psi(x,y,z):
    rho = norm(x,y,0)
    Psi = Psi_0 * (rho/r_jet(z))**2
#    Psi = np.pi*B_0(z)*r_core**2/(1-alpha(z)/2) * ((1 + rho**2/r_core**2)**(1 - alpha(z)/2) - 1)
    return Psi

def Partial_Derivative(func, input_values, var_index, h=delta):
    original_val = input_values[var_index]
    input_values[var_index] = original_val + h
    f_plus_h = func(*input_values)
    input_values[var_index] = original_val - h
    f_minus_h = func(*input_values)
    input_values[var_index] = original_val
    return (f_plus_h - f_minus_h) / (2 * h)

def Omega(x,y,z):
    Omega = Omega_0 * np.sqrt(1 - Psi(x, y, z)/Psi_0)
    return Omega

#Максимальный лоренц-фактор
def Gamma_max(x,y,z):
    Gamma_max = Gamma_in + 2*sigma_M*Psi(x, y, z)/Psi_0 * (1 - Psi(x, y, z)/Psi_0)
    return Gamma_max

def epsilon(x,y,z):
    rho = norm(x,y,0)
    epsilon = 1/2/(Gamma_in + sigma_M*rho/k(z))**2
#    epsilon = 1/sigma_M**2
    epsilon = 0
    return epsilon

def E_x(x,y,z):
    E_x = -Omega(x,y,z)/(2*np.pi) * Partial_Derivative(Psi,[x,y,z],0)
    return E_x

def E_y(x,y,z):
    E_y = -Omega(x,y,z)/(2*np.pi) * Partial_Derivative(Psi,[x,y,z],1)
    return E_y

def E_z(x,y,z):
    E_z = -Omega(x,y,z)/(2*np.pi) * Partial_Derivative(Psi,[x,y,z],2)
    return E_z

def B_p_x(x,y,z):
    rho = norm(x,y,0)
    B_p_x = -1/(2*np.pi*rho) * x/rho * Partial_Derivative(Psi,[x,y,z],2)
    return B_p_x

def B_phi_x(x,y,z):
    rho = norm(x,y,0)
    B_phi_x = (1+epsilon(x,y,z))*Omega(x,y,z)/(2*np.pi) * y/rho * norm(Partial_Derivative(Psi,[x,y,z],0), Partial_Derivative(Psi,[x,y,z],1), Partial_Derivative(Psi,[x,y,z],2))
    return B_phi_x

def B_x(x,y,z):
    return B_p_x(x,y,z) + B_phi_x(x,y,z)

def B_p_y(x,y,z):
    rho = norm(x,y,0)
    B_p_y = -1/(2*np.pi*rho) * y/rho * Partial_Derivative(Psi,[x,y,z],2)
    return B_p_y

def B_phi_y(x,y,z):
    rho = norm(x,y,0)
    B_phi_y = -(1+epsilon(x,y,z))*Omega(x,y,z)/(2*np.pi) * x/rho * norm(Partial_Derivative(Psi,[x,y,z],0), Partial_Derivative(Psi,[x,y,z],1), Partial_Derivative(Psi,[x,y,z],2))
    return B_phi_y

def B_y(x,y,z):
    return B_p_y(x,y,z) + B_phi_y(x,y,z)

def B_p_z(x,y,z):
    rho = norm(x,y,0)
    B_p_z = 1/(2*np.pi*rho) * (x/rho*Partial_Derivative(Psi,[x,y,z],0)+y/rho*Partial_Derivative(Psi,[x,y,z],1))
    return B_p_z

def B_z(x,y,z):
    return B_p_z(x,y,z)

#Формулы для компонент безразмерной гидродинамической (дрейфовой) скорости
def V_x(x,y,z):
    B = norm(B_x(x, y, z),B_y(x, y, z),B_z(x, y, z))
    V_x = (E_y(x, y, z)*B_z(x, y, z) - E_z(x, y, z)*B_y(x, y, z))/B**2
    return V_x

def V_y(x,y,z):
    B = norm(B_x(x, y, z),B_y(x, y, z),B_z(x, y, z))
    V_y = (E_z(x, y, z)*B_x(x, y, z) - E_x(x, y, z)*B_z(x, y, z))/B**2
    return V_y

def V_z(x,y,z):
    B = norm(B_x(x, y, z),B_y(x, y, z),B_z(x, y, z))
    V_z = (E_x(x, y, z)*B_y(x, y, z) - E_y(x, y, z)*B_x(x, y, z))/B**2
    return V_z

#Гидродинамический лоренц-фактор
def Gamma(x,y,z):
    V = norm(V_x(x, y, z),V_y(x, y, z),V_z(x, y, z))
    Gamma = 1/np.sqrt(1 - V**2)
    Gamma = np.sqrt(Gamma_in**2 - 1 + Gamma**2)
    return Gamma

#Доплер-фактор
def D(x,y,z): 
    D = 1/Gamma(x, y, z)/(1 - V_x(x, y, z)*n_x - V_y(x, y, z)*n_y - V_z(x, y, z)*n_z)
    return D

#Локальная замагниченность
def sigma(x,y,z):
    rho = norm(x,y,0)
    sigma = (1 - rho**2/r_jet(z)**2) * rho**2/r_jet(z)**2 * sigma_M/Gamma(x, y, z)
    return sigma

#Граница замагниченности
def x_mag(z):
    x_mag = r_jet(z)/np.sqrt(2) * np.sqrt(1 + np.sqrt(1 - 2*r_jet(z)**2/sigma_M**2 + 2*np.sqrt(r_jet(z)**4/sigma_M**4 + 4*Gamma_in**2/sigma_M**2)))
    return x_mag

def x_mag_2(z):
    x_mag = r_jet(z)/np.sqrt(2) * np.sqrt(1 + np.sqrt(1 - 2*r_jet(z)**2/sigma_M**2 - 2*np.sqrt(r_jet(z)**4/sigma_M**4 + 4*Gamma_in**2/sigma_M**2)))
    return x_mag

def x_mag_3(z):
    x_mag = r_jet(z)/np.sqrt(2) * np.sqrt(1 - np.sqrt(1 - 2*r_jet(z)**2/sigma_M**2 - 2*np.sqrt(r_jet(z)**4/sigma_M**4 + 4*Gamma_in**2/sigma_M**2)))
    return x_mag

#Аппроксимация внутренней конической границы
def x_approx(z):
    x_approx = sigma_M**(-1) * r_jet(z)**2
    return x_approx

#Заготовка для построения альфвеновской и быстрой магнитозвуковой поверхностей
def rounded_plateau_function(x, x_start, x_end, height, radius):
    """
    Функция, возвращающая значение y для прямоугольного плато с закруглёнными углами.
    """
    width = x_end - x_start
    radius = min(radius, width / 2)
    
    y = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        if xi < x_start or xi > x_end:
            y[i] = 0
        elif xi < x_start + radius:
            # Левый закруглённый угол (четверть окружности)
            dx = xi - (x_start + radius)
            y[i] = height - radius + np.sqrt(max(0, radius**2 - dx**2))
        elif xi <= x_end - radius:
            # Плато
            y[i] = height
        else:
            # Правый закруглённый угол (четверть окружности)
            dx = xi - (x_end - radius)
            y[i] = height - radius + np.sqrt(max(0, radius**2 - dx**2))
    
    return y

def Pseudo_Surface(x,x_start,x_end,height):
    l = np.abs(x_end - 0)
    z = height * (1 - x**2/l**2)**(1/3)
    return z

z_min = 1
z_max = z_min + 7300
z_0 = 100
x_0 = 0
y_0 = 0
#Построим равномерную квадратную сетку NxN, внутри которой лежит джет
N = 10000
Z = np.linspace(z_min, z_max,N)
X_mag_data_1 = []
X_mag_data_2 = []
X_mag_data_3 = []
Z_1 = []
Z_23 = []
z_sigma_M = z_max
z_sqrt_sigma_M = 100
for i in range(N):
    z_0 = Z[i]
    if (r_jet(z_max) - x_mag(z_0)) >= 0:
        Z_1 = np.append(Z_1,z_0)
        X_mag_data_1 = np.append(X_mag_data_1,x_mag(z_0))
        if (sigma_M - np.sqrt(4*r_jet(z_0)**2 + 16*Gamma_in**2)) >= 0:
            Z_23 = np.append(Z_23,z_0)
            X_mag_data_2 = np.append(X_mag_data_2,x_mag_2(z_0))
            X_mag_data_3 = np.append(X_mag_data_3,x_mag_3(z_0))
    if (z_max >= 7000):
        if np.abs(r_jet(z_0) - np.sqrt(sigma_M)) < delta:
            z_sqrt_sigma_M = z_0
        if np.abs(r_jet(z_0) - sigma_M) < delta:
            z_sigma_M = z_0
    else:
        if np.abs(r_jet(z_0) - np.sqrt(sigma_M)) < delta:
            z_sqrt_sigma_M = z_0

#Быстрая магнитозвуковая поверхность
x_fm = Gamma_in**(-1/2) * sigma_M**(1/2)
z_fm = sigma_M/Gamma_in 
#Параметры для быстрой магнитозвуковой поверхности
x_start, x_end = -x_fm, x_fm
height = z_fm
x_for_fm = np.linspace(x_start, x_end, N)
z_for_fm = Pseudo_Surface(x_for_fm, x_start, x_end, height)

print(z_sigma_M)
print(z_sqrt_sigma_M)
print(z_fm)

#Альфвеновская поверхность
x_a = 1 
z_a = z_fm
#Параметры для альфвеновской поверхности
x_start, x_end = -x_a, x_a
height = z_a
x_for_a = np.linspace(x_start, x_end, N)
z_for_a = Pseudo_Surface(x_for_a, x_start, x_end, height)

#Схематичный рисунок границ сильно- слабозамагниченных течений
plt.figure(figsize=(6, 9))
#Добавляем быструю магнитозвуковую поверхность
plt.plot(x_for_fm, z_for_fm, color='black', linewidth=3)
#Добавляем альфвеновскую поверхность
plt.plot(x_for_a, z_for_a, color='black', linewidth=3)
# Добавляем вертикальную линию в точке x=0
plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
plt.plot(r_jet(Z),Z,color = 'black', linewidth = 2, linestyle = '--')
plt.plot(-r_jet(Z),Z,color = 'black', linewidth = 2, linestyle = '--')
plt.plot(x_approx(Z),Z,color = 'black', linewidth = 2, linestyle = '--')
plt.plot(-x_approx(Z),Z,color = 'black', linewidth = 2, linestyle = '--')
if (z_max >= 7000):
    plt.annotate('', xy=(sigma_M, z_sigma_M), xytext=(0, z_sigma_M),arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    plt.annotate('', xy=(np.sqrt(sigma_M), z_sqrt_sigma_M), xytext=(0, z_sqrt_sigma_M),arrowprops=dict(arrowstyle='<->', color='black', lw=2))
else:
    plt.annotate('', xy=(np.sqrt(sigma_M), z_sqrt_sigma_M), xytext=(0, z_sqrt_sigma_M),arrowprops=dict(arrowstyle='<->', color='black', lw=2))
plt.plot(X_mag_data_1,Z_1,color = 'black', linewidth = 2, linestyle = '-')
plt.plot(X_mag_data_2,Z_23,color = 'black', linewidth = 2, linestyle = '-')  
plt.plot(X_mag_data_3,Z_23,color = 'black', linewidth = 2, linestyle = '-') 
plt.plot(-X_mag_data_1,Z_1,color = 'black', linewidth = 2, linestyle = '-')
plt.plot(-X_mag_data_2,Z_23,color = 'black', linewidth = 2, linestyle = '-')  
plt.plot(-X_mag_data_3,Z_23,color = 'black', linewidth = 2, linestyle = '-')
# Добавление простого текста
if z_max >= 7000:
    plt.text(25, 6950, '$\sigma_{M}$', fontsize=22, color='black')
    plt.text(-90, 200, 'a)', fontsize=26, color='black')
else:
    plt.text(5, 75, '$\sigma_{M}$', fontsize=22, color='black')
    plt.text(-6, 42.5, 'F', fontsize=22, color='black')
    plt.text(-3, 25, 'A', fontsize=22, color='black')
    plt.text(-16.5, 5, 'b)', fontsize=26, color='black')
plt.xlabel('$x$', fontsize=15)
plt.ylabel('$z/R_{L}$', fontsize=15)
ax = plt.gca()
#ax.xaxis.set_major_locator(MaxNLocator(5))  # максимум 5 делений
ax.yaxis.set_major_locator(MaxNLocator(5))  # максимум 5 делений
plt.ylim(0,z_max)
if (z_max >= 7000):
    plt.xlim(-sigma_M,sigma_M)
plt.title('$\sigma_{M}=100, \Gamma_{in}=2, z_{st}/R_{L}=1$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.show()

end_time = time.time()  # время окончания выполнения
execution_time = end_time - start_time  # вычисляем время выполнения
 
print(f"Время выполнения программы: {execution_time} секунд")