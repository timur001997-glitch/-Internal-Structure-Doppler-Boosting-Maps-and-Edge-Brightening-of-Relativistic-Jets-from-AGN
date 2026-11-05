import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import multiprocessing
from numba import jit

start_time = time.time()  # время начала выполнения

sigma_M = 5 #параметр замагниченности Майкеля
Gamma_0 = 5.0 #гидродинамический лоренц-фактор на оси джета
h_0 = 1*10**(2) #магнитное поле на расстоянии одного R_L от центральной машины в Гс
c = 3*10**10 #скорость света в см/c
G = 6.67*10**(-8) #гравитационная постоянная в системе СГС
M = 1*10**5 * 2*10**33 #масса компактного объекта в г
#M = 2*10**33 #масса Солнца в г
r_g = 2*G*M/c**2 #гравитационный радиус в см
R_L = 10*r_g #радиус светового цилиндра в см
R_L = 10**(16) #уже учтён при переходе в лабораторную СО в формуле интенсивности
m_e = 9.1*10**(-28) #масса электрона в г
q = 4.8 * 10**(-10) #заряд электрона в системе СГС
t_0 = R_L/c #обезразмеривание времени

P_0 = 4.5 * 10**(-10) #давление в дин/см^2 для M87
z_00 = 220 * 3.086 * 10**18 #характерное расстояние вдоль оси джета в см для M87
z_00 = z_00/10**16 #характерное безразмерное расстояние вдоль оси джета для M87
Psi_0 = np.pi #полный магнитный поток!
r_jet_cr = np.sqrt(Gamma_0*sigma_M) #в световых цилиндрах R_L (безразмерная)
r_core = 1 #в световых цилиндрах R_L (безразмерная)

delta = 10**(-4) #анти-поломка программы - малая добавка delta

print(r_jet_cr)

Omega_0 = 1

k_B = 1.38*10**(-16) #постоянная Больцмана в СГС
lambda_gamma = 10**10 #параметр множественности излучающей плазмы
lambda_e = 10**12 #параметр множественности фоновой плазмы
nu = 230*10**9 #частота радиоизлучения в Гц
p = 2.5 #спектральный индекс энергетического распределения излучающих частиц
Theta = np.pi*50/180 #угол между осью z вращения джета и лучом зрения
Theta_deg = int(round(np.degrees(Theta)))  #угол между осью z вращения джета и лучом зрения в градусах
theta = np.pi*3/180 #угол полураскрыва конического джета
z_redshift = 0 #красное смещение

#Характеристика параболической формы джета
C_shape = (Psi_0**2*h_0**2/(8*np.pi**3*P_0*z_00**2))**(1/4)

#Направление луча зрения в лабораторной СО
n_x = np.sin(Theta)
n_y = 0
n_z = np.cos(Theta)

#Вычислим константу в излучательной способности a_nu
kappa_nu = np.sqrt(3/4) * gamma((3*p-1)/(12)) * gamma((3*p+7)/(12)) * q**3/m_e/c**2 * ((3*q)/(2*np.pi*m_e*c))**((p-1)/2) * nu**(-(p-1)/2)

#Вычислим константу перед интегралом в яркостной температуре
R = 3**(p/2)/(8*(2*np.pi)**((p+1)/2)) * (p-1)*(p+7/3)/(p+1) * gamma((3*p-1)/12) * gamma((3*p+7)/12)
T_0 = lambda_gamma * m_e*c**2/k_B * R * (q*h_0/(m_e*c))**((p+3)/2) * nu**(-(p+3)/2)

#Вычислим константу в формуле связи энергетики излучающих частиц с МГД
Sigma = q*h_0*R_L/2/lambda_gamma/m_e/c**2
print(Sigma)

#Формулы для ЭМ полей через магнитный поток
@jit(nopython=True)
def norm(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return r

#Геометрическая форма джета r_jet
@jit(nopython=True)
def r_jet(z):
    r_jet = (Psi_0**2*h_0**2/(8*np.pi**3*P_0*z_00**2))**(1/4) * z**(2/4)
#    r_jet = np.tan(theta) * z
    return r_jet #форма джета в единицах R_L (безразмерная)

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

@jit(nopython=True)
def Partial_Derivative(func, x,y,z, var_index, h=delta):  
    # Вычисляем значения со смещением
    if var_index == 0:  # производная по x
        f_plus = func(x + h, y, z)
        f_minus = func(x - h, y, z)
    elif var_index == 1:  # производная по y
        f_plus = func(x, y + h, z)
        f_minus = func(x, y - h, z)
    else:  # производная по z
        f_plus = func(x, y, z + h)
        f_minus = func(x, y, z - h)  
    # Центральная разность (наиболее точная)
    return (f_plus - f_minus) / (2 * h)

@jit(nopython=True)
def Omega(x,y,z):
    Omega = Omega_0 * np.sqrt(1 - Psi(x, y, z)/Psi_0)
#    Omega = Omega_0
    return Omega

#Максимальный лоренц-фактор
@jit(nopython=True)
def Gamma_max(x,y,z):
    Gamma_max = Gamma_0 + 2*sigma_M*Psi(x, y, z)/Psi_0 * (1 - Psi(x, y, z)/Psi_0)
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
#    B_p_z = B_0(z) * (1 + rho**2/r_core**2)**(-alpha(z)/2)
    return B_p_z

@jit(nopython=True)
def B_z(x,y,z):
    return B_p_z(x,y,z)

#Формулы для компонент гидродинамической (дрейфовой) скорости
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

#Гидродинамическая (дрейфовая) скорость безразмерная
@jit(nopython=True)
def V_dr(x,y,z):
    V = norm(V_x(x, y, z),V_y(x, y, z),V_z(x, y, z))
    V_dr = V
    return V_dr

#Гидродинамический лоренц-фактор
@jit(nopython=True)
def Gamma(x,y,z):
    V = norm(V_x(x, y, z),V_y(x, y, z),V_z(x, y, z))
    Gamma = 1/np.sqrt(1 - V**2)
    Gamma = np.sqrt(Gamma_0**2 - 1 + Gamma**2)
    return Gamma

#Доплер-фактор
@jit(nopython=True)
def D(x,y,z): 
    D = 1/Gamma(x, y, z)/(1 - V_x(x, y, z)*n_x - V_y(x, y, z)*n_y - V_z(x, y, z)*n_z)
    return D

#ЭМ лоренц-инвариант
@jit(nopython=True)
def h(x,y,z):
    B = norm(B_x(x, y, z),B_y(x, y, z),B_z(x, y, z))
    E = norm(E_x(x, y, z),E_y(x, y, z),E_z(x, y, z))
    h = np.sqrt(B**2 - E**2)
    return h

#Косинус угла между магнитным полем и лучом зрения в лабораторной СО
@jit(nopython=True)
def Cos_chi_lab(x,y,z):
    B = norm(B_x(x, y, z),B_y(x, y, z),B_z(x, y, z))
    Cos_chi = (B_x(x, y, z)*n_x + B_y(x, y, z)*n_y + B_z(x, y, z)*n_z)/B
    return Cos_chi

#Косинус угла между магнитным полем и лучом зрения в дрейфовой СО
@jit(nopython=True)
def Cos_chi(x,y,z):
    Cos_chi = D(x, y, z) * Cos_chi_lab(x, y, z)
    return Cos_chi

#Минимальный Лоренц-фактор, связь энергетики излучающих частиц с МГД
@jit(nopython=True)
def gamma_min(x,y,z):
    B = norm(B_x(x,y,z), B_y(x,y,z), B_z(x,y,z))
    B_p = norm(B_p_x(x,y,z),B_p_y(x,y,z),B_p_z(x, y, z))
    gamma_min_0 = 10**2
    gamma_min = gamma_min_0 + (p-2)/(p-1) * V_dr(x, y, z)/Gamma(x, y, z)/(1+V_dr(x,y,z)**2/3) * (Sigma * B**2/B_p - 1 * lambda_e/lambda_gamma*Gamma(x,y,z))
#    gamma_min = gamma_min_0
    return gamma_min

#Проведём все вычисления через координату x. Тогда
#Уравнение луча зрения (line of sight)
@jit(nopython=True)
def z_los(x, x_0, y_0, z_0):
    z_los = z_0 + 1/np.tan(Theta)*(x - x_0)
    return z_los

#Границы по координате x для луча зрения внутри джета
@jit(nopython=True)
def x_1(x_0,y_0,z_0):
    x_1 = (C_shape**2 * 1/np.tan(Theta) - np.sqrt((C_shape**2 * 1/np.tan(Theta))**2 + 4*(C_shape**2*z_0 - y_0**2)))/2
    return x_1

@jit(nopython=True)
def x_2(x_0,y_0,z_0):
    x_2 = (C_shape**2 * 1/np.tan(Theta) + np.sqrt((C_shape**2 * 1/np.tan(Theta))**2 + 4*(C_shape**2*z_0 - y_0**2)))/2
    return x_2

#Безразмерная подынтегральная функция в формуле яркостной температуры
@jit(nopython=True)
def I(x, x_0, y_0, z_0):
    y = y_0
    z = z_los(x, x_0, y_0, z_0)
    B_p = norm(B_p_x(x,y,z),B_p_y(x,y,z),B_p_z(x, y, z))
    I = D(x,y,z)**(2+(p-1)/2) * (h(x,y,z))**((p+1)/2) * np.sqrt(1+1/np.tan(Theta)**2) * np.sqrt(1 - Cos_chi(x,y,z)**2)**((p+1)/2) * B_p * gamma_min(x, y, z)**(p-1)/Gamma(x,y,z)
    return I

#Яркостная температура
def T_br(x_0,y_0,z_0):
    A = x_1(x_0, y_0, z_0)
    B = x_2(x_0, y_0, z_0)
    II = quad(I, A, B, args=(x_0, y_0, z_0))
    T_br = T_0 * II[0]
    return T_br

#Обёртка для одного узла сетки (аналог Solver_gamma_com из предыдущего кода)
def Solver_point(args):
    i, j, z_0, y_0 = args
    x_0 = 0
    if np.sqrt(x_0**2 + y_0**2) <= r_jet(z_0):
        T = T_br(x_0, y_0, z_0)
        gm = gamma_min(x_0, y_0, z_0)
    else:
        T = 1
        gm = 0
    return T, gm

if __name__ == '__main__':
    z_min = 10
    z_max = z_min + 1000*3
    N = 1000
    Z = np.linspace(z_min, z_max,N)
    Y = np.linspace(-r_jet(z_max), r_jet(z_max),N)

    #Сформируем список заданий для воркеров: (i, j, z_0, y_0)
    args_list = []
    for i in range(N):
        z_0 = Z[i]
        for j in range(N):
            y_0 = Y[j]
            args_list.append((i, j, z_0, y_0))

    n_proc = int(multiprocessing.cpu_count()*6/8)
    print('n_proc =', n_proc)

    with multiprocessing.Pool(n_proc) as pp:
        results = pp.map(Solver_point, args_list)

    #Собираем результаты обратно в 2D-сетку
    T_data = np.zeros((N, N))
    gamma_min_data = []
    for idx, (T, gm) in enumerate(results):
        i = idx // N
        j = idx % N
        T_data[i, j] = T
        if gm != 0:
            gamma_min_data.append(gm)

    print(min(gamma_min_data))
    print(max(gamma_min_data))
    print(T_data.max())
    T_data = np.transpose(T_data)
    Y = np.flip(Y)
    #T_data = np.flip(T_data)

    z_min, z_max = min(Z), max(Z)
    y_min, y_max = min(Y), max(Y)
    T_min, T_max = T_data.min(),T_data.max()

    plt.figure(figsize=(8, 4.5))
    my_norm = colors.LogNorm(vmin=10**6, vmax=T_max)
    im = plt.imshow(T_data,origin='lower',cmap='afmhot',extent=[z_min, z_max, y_min, y_max],aspect="auto",norm=my_norm)
    # Добавляем шкалу интенсивности (colorbar)
    cbar = plt.colorbar(im)
    cbar.ax.set_title('$T_{br}, K$', fontsize=15)
    cbar.ax.tick_params(which='major',labelsize=14)
    plt.xlabel('$z/R_{L}$', fontsize=15)
    plt.ylabel('$y/R_{L}$', fontsize=15)
#    plt.title('$\sigma_{M}=100, \Gamma_{in}=1, z_{st}/R_{L}=100, \\Theta = {50}^{\circ}$',fontsize=15)
    plt.title(f'$\\sigma_{{M}}={sigma_M}, \\Gamma_{{in}}={Gamma_0:g}, z_{{st}}/R_{{L}}={z_min:g}, \\Theta = {Theta_deg}^{{\\circ}}$', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=14)
#    plt.ylim(-100, 100)
    plt.show()

    end_time = time.time()  # время окончания выполнения
    execution_time = end_time - start_time  # вычисляем время выполнения

    print(f"Время выполнения программы: {execution_time} секунд")