# Po-Chen Kuo
# UW Neuro 545 final project
# Mar 17, 2022


# import
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#################################################################
### Part I: 1-D ring model for nonlinear summation
#################################################################

# parameters
tau_E, tau_I = 20*0.001, 10*0.001
N = 180
J_EE, J_IE, J_EI, J_II = 0.044, 0.042, 0.023, 0.018
alpha = 2.0
k = 0.04
sigma_FF = 30
sigma_ori = 32

# set up 1-D ring SSN model
def d_theta(theta1, theta2):
    if theta1 < theta2:
        _theta = theta1
        theta1 = theta2
        theta2 = _theta

    if theta1-theta2 <= 180:
        delta_theta = theta1-theta2
    else:
        delta_theta = 360-(theta1-theta2)

    return delta_theta


def gaussian_theta(theta1, theta2, sigma):
    delta = d_theta(theta1, theta2)

    return np.exp(- delta**2/ (2*sigma**2))


def gen_connection(theta1, theta2, sigma=sigma_ori):
    W = []
    for J in [J_EE, J_IE, J_EI, J_II]:
        W.append(J * gaussian_theta(theta1, theta2, sigma))
    
    return W


# external input
def h_theta(theta_pre, theta_stim, sigma=sigma_FF):
    return gaussian_theta(theta_pre, theta_stim, sigma)


# unit input
def I_E(theta_pre, theta_stim_list, r_E_t, r_I_t, c):
    ext_input = 0
    for theta_stim in theta_stim_list:
        ext_input += c * h_theta(theta_pre, theta_stim)
    
    unit_input = 0
    for i, theta_pre_prime in enumerate(preferred_thetas):
        W = gen_connection(theta_pre, theta_pre_prime)
        unit_input += W[0] * r_E_t[i]
        unit_input -= W[2] * r_I_t[i]
    
    return ext_input + unit_input


def I_I(theta_pre, theta_stim_list, r_E_t, r_I_t, c):
    ext_input = 0
    for theta_stim in theta_stim_list:
        ext_input += c * h_theta(theta_pre, theta_stim)
    
    unit_input = 0
    for i, theta_pre_prime in enumerate(preferred_thetas):
        W = gen_connection(theta_pre, theta_pre_prime)
        unit_input += W[1] * r_E_t[i]
        unit_input -= W[3] * r_I_t[i]
    
    return ext_input + unit_input


# steady state firing
def r_steday_state(total_input, alpha=alpha, k=k):
    return k * np.power(np.max(total_input, 0), 2.0)

# dynamics
def dr_Edt(i, theta_pre, theta_stim_list, r_E_t, r_I_t, c):
    return 1/ tau_E * (-r_E_t[i] + r_steday_state(I_E(theta_pre, theta_stim_list, r_E_t, r_I_t, c)))


def dr_Idt(i, theta_pre, theta_stim_list, r_E_t, r_I_t, c):
    return 1/ tau_I * (-r_I_t[i] + r_steday_state(I_I(theta_pre, theta_stim_list, r_E_t, r_I_t, c)))


# main simulation function
def sim_1d_ring(theta_stim_list):
    r_E = np.ones(180).reshape(180, 1)
    r_I = np.ones(180).reshape(180, 1)
    time_step = 0.001
    time_points = 40

    for t in range(time_points):
        r_E_tp1 = []
        r_I_tp1 = []
        print(t)
        
        for i, theta_pre in enumerate(preferred_thetas):
            r_E_tp1.append(dr_Edt(i, theta_pre, theta_stim_list, r_E[:, -1], r_I[:, -1], 200) * time_step)
            r_I_tp1.append(dr_Idt(i, theta_pre, theta_stim_list, r_E[:, -1], r_I[:, -1], 200) * time_step)
        r_E_tp1 = np.array(r_E_tp1).reshape(180, 1)
        r_I_tp1 = np.array(r_I_tp1).reshape(180, 1)

        r_E = np.concatenate((r_E, r_E_tp1), axis=1)
        r_I = np.concatenate((r_I, r_I_tp1), axis=1)
    
    return r_E, r_I


# run simulation for stimuli at 45, 135, and both
preferred_thetas = np.arange(1, 181)

theta_stim_list = [45]
r_E_45, r_I_45 = sim_1d_ring(theta_stim_list)

theta_stim_list = [135]
r_E_135, r_I_135 = sim_1d_ring(theta_stim_list)

theta_stim_list = [45, 135]
r_E_both, r_I_both = sim_1d_ring(theta_stim_list)


# plot the result
plt.figure(dpi=300)
t = 30
plt.plot(r_E_45[:, t], label='stimulus at 45')
plt.plot(r_E_135[:, t], label='stimulus at 135')
plt.plot(r_E_both[:, t], label='stimuli at 45 & 135')
plt.plot(r_E_45[:, t]+r_E_135[:, t], linestyle='--', label='sum')
plt.plot(np.average([r_E_45[:, t], r_E_135[:, t]], axis=0), linestyle='--', label='average')
plt.title('Excitatory firing rate')
plt.ylabel('firing rate')
plt.xlabel('preferred orientation')
plt.legend(bbox_to_anchor=(1, 0.5))
plt.show()

plt.figure(dpi=300)
t = 30
plt.plot(r_I_45[:, t], label='stimulus at 45')
plt.plot(r_I_135[:, t], label='stimulus at 135')
plt.plot(r_I_both[:, t], label='stimuli at 45 & 135')
plt.plot(r_I_45[:, t]+r_I_135[:, t], linestyle='--', label='sum')
plt.plot(np.average([r_I_45[:, t], r_I_135[:, t]], axis=0), linestyle='--', label='average')
plt.title('Inhibitory firing rate')
plt.ylabel('firing rate')
plt.xlabel('preferred orientation')
plt.legend(bbox_to_anchor=(1, 0.5))
plt.show()



#################################################################
### Part II: 1-D linear model for surround suppression
#################################################################

# parameters
tau_E, tau_I = 20*0.001, 10*0.001
N = 101
delta_x = float(1)/3
sigma_RF = 0.125 * delta_x
J_EE, J_IE, = 1.0, 1.25
W_EI, W_II = 1.0, 0.75
alpha = 2.0
k = 0.01
sigma_EE = float(2)/3
sigma_IE = float(4)/3


# set up 1-D linear model
def d_x(x1, x2):
    return np.abs(x1-x2)


def gaussian_x(x1, x2, sigma):
    delta = d_x(x1, x2)
    return np.exp(- delta**2/ (2*sigma**2))


def gen_connection_x(x1, x2):
    W = []
    W.append(J_EE * gaussian_x(x1, x2, sigma_EE))
    W.append(J_IE * gaussian_x(x1, x2, sigma_IE))
    W.append(W_EI)
    W.append(W_II)
    return W


# external input
def h_x(x, stim_l):
    return (1/(1+np.exp((-x-stim_l/2.0)/ sigma_RF))) * (1 - (1/(1+np.exp((-x+stim_l/2.0)/ sigma_RF))))


# unit input
def I_E(x, stim_l, r_E_t, r_I_t, c):
    
    ext_input = c * h_x(x, stim_l)
    
    unit_input = 0
    for i, x_prime in enumerate(xs):
        W = gen_connection_x(x, x_prime)
        unit_input += W[0] * r_E_t[i]
        if x_prime == x:
            # print('inh project: {}'.format(x_prime))
            unit_input -= W[2] * r_I_t[i]
    
    return ext_input + unit_input


def I_I(x, stim_l, r_E_t, r_I_t, c):
    ext_input = c * h_x(x, stim_l)
    
    unit_input = 0
    for i, x_prime in enumerate(xs):
        W = gen_connection_x(x, x_prime)
        unit_input += W[1] * r_E_t[i]
        if x_prime == x:
            # print('inh project: {}'.format(x_prime))
            unit_input -= W[3] * r_I_t[i]
    
    return ext_input + unit_input


# steday state firing
def r_steday_state(total_input, alpha=alpha, k=k):
    return k * np.power(np.max(total_input, 0), alpha)  
    # return total_input   # use linear for figure 2

# dynamics
def dr_Edt(i, x, stim_l, r_E_t, r_I_t, c):
    return 1/ tau_E * (-r_E_t[i] + r_steday_state(I_E(x, stim_l, r_E_t, r_I_t, c)))


def dr_Idt(i, x, stim_l, r_E_t, r_I_t, c):
    return 1/ tau_I * (-r_I_t[i] + r_steday_state(I_I(x, stim_l, r_E_t, r_I_t, c)))


# main simulation function
def sim_1d_line(stim_l):
    r_E = np.ones(N).reshape(N, 1)
    r_I = np.ones(N).reshape(N, 1)
    time_step = 0.001
    time_points = 40

    for t in range(time_points):
        r_E_tp1 = []
        r_I_tp1 = []
        print(t)
        
        for i, x in enumerate(xs):
            r_E_tp1.append(dr_Edt(i, x, stim_l, r_E[:, -1], r_I[:, -1], 241) * time_step)
            r_I_tp1.append(dr_Idt(i, x, stim_l, r_E[:, -1], r_I[:, -1], 241) * time_step)
        r_E_tp1 = np.array(r_E_tp1).reshape(N, 1)
        r_I_tp1 = np.array(r_I_tp1).reshape(N, 1)

        r_E = np.concatenate((r_E, r_E_tp1), axis=1)
        r_I = np.concatenate((r_I, r_I_tp1), axis=1)
        print(r_E_tp1[int((N-1)/2)])
    
    return r_E, r_I


# run simulation
xs = np.linspace(-50.0/3, +50.0/3, N)

stim_l = 1
r_E_1, r_I_1 = sim_1d_line(stim_l)

stim_l = 5
r_E_5, r_I_5 = sim_1d_line(stim_l)

stim_l = 7
r_E_7, r_I_7 = sim_1d_line(stim_l)


# plot the result
plt.figure(dpi=300)
t = 30
plt.plot(xs, r_E_1[:, t], label='stimulus of length 1')
plt.plot(xs, r_E_5[:, t], label='stimulus of length 5')
plt.plot(xs, r_E_7[:, t], label='stimulus of length 7')
plt.title('Excitatory firing rate')
plt.ylabel('firing rate')
plt.xlabel('grid position')
plt.xlim((-8, 8))
plt.legend(bbox_to_anchor=(1, 0.5))
plt.show()

plt.figure(dpi=300)
t = 30
plt.plot(xs, r_I_1[:, t], label='stimulus of length 1')
plt.plot(xs, r_I_5[:, t], label='stimulus of length 5')
plt.plot(xs, r_I_7[:, t], label='stimulus of length 7')
plt.title('Inhibitory firing rate')
plt.ylabel('firing rate')
plt.xlabel('grid position')
plt.xlim((-8, 8))
plt.legend(bbox_to_anchor=(1, 0.5))
plt.show()



#################################################################
### Part III: Phase space analysis of SSN
#################################################################

# paramets: uncomment the proper setting to use

# single steady state
# tau_E, tau_I = 1.0, 1.0
# J_EE, J_EI, J_IE, J_II = 1.1, 0.9, 2, 1
# alpha_E, alpha_I = 3, 3
# g_E, g_I = 0.4, 0.3
# r0_E, r0_I = 0.3, 0.1
# r0 = np.array([r0_E, r0_I])

# 2 steady states
# tau_E, tau_I = 1.0, 1.0
# J_EE, J_EI, J_IE, J_II = 1.5, 1.0, 0.5, 1.0
# alpha_E, alpha_I = 3, 3
# g_E, g_I = 0.1, 0.1
# r0_E, r0_I = 0.35, 0.5
# r0 = np.array([r0_E, r0_I])

# limit cycle A
tau_E, tau_I = 0.1, 1.0
J_EE, J_EI, J_IE, J_II = 1.5, 1.0, 10.0, 1.0
alpha_E, alpha_I = 3, 3
g_E, g_I = 0.7, 0.01
r0_E, r0_I = 0.3, 0.1
r0 = np.array([r0_E, r0_I])

# limit cycle B
# tau_E, tau_I = 0.1, 1.0
# J_EE, J_EI, J_IE, J_II = 1.5, 1.0, 10.0, 1.0
# alpha_E, alpha_I = 3, 3
# g_E, g_I = 5, 0.01
# r0_E, r0_I = 0.69, 5.15
# r0 = np.array([r0_E, r0_I])


# dynamics equation
def ss_firing(r, alpha):
    r = np.max(r, 0)
    r = np.power(r, alpha)
    return r


def dr_Edt(r_E, r_I, tau_E, J_EE, J_EI, g_E, alpha_E):
    return 1/ tau_E * (-r_E + ss_firing(J_EE*r_E - J_EI*r_I + g_E, alpha_E))


def dr_Idt(r_E, r_I, tau_I, J_IE, J_II, g_I, alpha_I):
    return 1/ tau_I * (-r_I + ss_firing(J_IE*r_E - J_II*r_I + g_I, alpha_I))


def pend(r, t, tau_E, tau_I, J_EE, J_EI, J_IE, J_II, alpha_E, alpha_I, g_E, g_I):
    r_E, r_I = r
    drdt = [
        dr_Edt(r_E, r_I, tau_E, J_EE, J_EI, g_E, alpha_E),
        dr_Idt(r_E, r_I, tau_I, J_IE, J_II, g_I, alpha_I)
    ]
    return drdt

# solve for trajectory
delta_t = 3
res = 200
t = np.linspace(0, delta_t, delta_t*res+1)

sol= odeint(pend, r0, t, args=(tau_E, tau_I, J_EE, J_EI, J_IE, J_II, alpha_E, alpha_I, g_E, g_I))
x, y = sol.T

# plot the temporal space dynamics and phase space analysis
fig = plt.figure(figsize=(15,5), dpi=300)
fig.subplots_adjust(wspace = 0.5, hspace = 0.3)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# temporal space
ax1.plot(t, x, 'r-', label='r_E(t)')
ax1.plot(t, y, 'b-', label='r_I(t)')
ax1.set_title("Dynamics in time")
ax1.set_xlabel("time")
ax1.set_ylabel('firing rate')
ax1.legend(loc='best')
ax1.set_xlim(0, 3)

# phase space
ax2.quiver(sol[:, 0][:-1], sol[:, 1][:-1], sol[:, 0][1:]-sol[:, 0][:-1], sol[:, 1][1:]-sol[:, 1][:-1], scale_units='xy', angles='xy', scale=1)
ax2.set_xlabel("Excitatory firing rate")
ax2.set_ylabel("Inhibitory firing rate")
ax2.set_title("Phase space")
ax2.legend()
ax2.set_xlim(-0.2, 0.5)
ax2.set_ylim(0, 1.2)

plt.show()