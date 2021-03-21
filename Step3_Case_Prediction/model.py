'''SEIR model to assess the spread of COVID-19'''
'''Updated on 2021.01.23'''
''' Data
Mobility(transit): 2020.02.15 ~ 2020.12.28 (16 days delayed)
Temperature(@Tokyo, no data for Japan): 2020.02.19 ~ 2020.12.31 (12 days delayed)
Infectious: 2020.03.02 ~ 2021.01.12
Removed: 2020.03.02 ~ 2021.01.12
Inf_new: 2020.03.02 ~ 2021.01.12
Reproduction number: 2020.03.02 ~ 2021.01.12'''
'''Fitting period, 1: ~ 2020.03.23 (22 days), 2: ~ 2020.06.30 (121 days), 3: ~ 2020.11.10 (254 days), 4: ~ 2021.01.12 (317 days)'''

'''Calling library'''
import numpy as np
import pandas as pd

'''Functions'''
# Susceptible
def f_sus(t_cou, sus_old, exp_old, inf_old, rem_old, b, a, r):
    x = - b[i] * sus_old * inf_old
    return x

# Exposed
def f_exp(t_cou, sus_old, exp_old, inf_old, rem_old, b, a, r):
    x = b[i] * sus_old * inf_old - a * exp_old
    return x

# Infectious
def f_inf(t_cou, sus_old, exp_old, inf_old, rem_old, b, a, r): 
    x = a * exp_old - r * inf_old
    return x

# Recovered + Death
def f_rem(t_cou, sus_old, exp_old, inf_old, rem_old, b, a, r):
    x = r * inf_old
    return x

'''Main'''
# Initial conditions
a = 0.40 #0.244368099 #0.14463166
r = 0.0696911140550688 #0.0295524227 #0.14799941
#t_delay = 8
n_total = 1.382 * 1.0e+7
exdata_I_new = np.loadtxt("I_new.csv") # New infected peoplw
exp_ini = (exdata_I_new[0]+exdata_I_new[1])/(2*a) #30730.71429 #37.85714 #0.0 #29735.0/n_total #110.0 / n_total
inf_ini = 103 #0.142857143 #58923.14286 #24.28571 #63620.0/n_total #/ n_total
rem_ini = 0.0 #225416.2857 #12.85714 #0.0 #232854.143/n_total
sus_ini = n_total - exp_ini - inf_ini -rem_ini #1.0 #0.959
exdata_reproduction = np.loadtxt("beta_Data.csv") # unit is %
n_b = 311 #number of beta
n_e = 313 #number of env
n1 = 206 #number of data till Oct 15


for t_delay2 in range(5):
    t_delay = t_delay2 + 6
    a_file = 'beta_pred_' + str(t_delay) + '.csv'
    data_beta = pd.read_csv(a_file)
    clm = np.arange(0,n_b-n1)
    ind = np.arange(0,n_b+t_delay)
    NewI = pd.DataFrame(index=ind, columns=clm)

    for t1 in range(n_b-n1-1) :
        sus_old = sus_ini
        exp_old = exp_ini
        inf_old = inf_ini
        rem_old = rem_ini
        
        # Estimated paramters
        n = (n1 + t_delay + t1) * 10
        rt = np.zeros(n)
        # mobility = np.zeros(n)
        # temp = np.zeros(n)
        # temp_ratio = np.zeros(n)
        b = np.zeros(n)
        y = 0
        for i in range(n):
            if i % 10 == 0:
                y += 1
            if i < (n1 + t1) *10:
                rt[i] = exdata_reproduction[y - 1]/n_total
            else:
                rt[i] = data_beta.iloc[y-1,t1+1]/n_total
    
    
        # Time 
        t_cou = 0.0
        t_tot = float(n) / 10.0 # [day]
        dt = 1.0 / 10.0 # [day]
    
        # Matrix for output
        sus = np.zeros(n+1)
        exp = np.zeros(n+1)
        inf = np.zeros(n+1)
        inf_new = np.zeros(n+1)
        rem = np.zeros(n+1)
        sus_cal = np.zeros(int(float(n) / 10.0)+1)
        exp_cal = np.zeros(int(float(n) / 10.0)+1)
        inf_cal = np.zeros(int(float(n) / 10.0)+1)
        rem_cal = np.zeros(int(float(n) / 10.0)+1)
        inf_tot_cal = np.zeros(int(float(n) / 10.0)+1)
        inf_new_cal = np.zeros(int(float(n) / 10.0)+1)
        inf_exp = np.zeros(int(float(n) / 10.0)+1)
        rem_exp = np.zeros(int(float(n) / 10.0)+1)
        inf_new_exp = np.zeros(int(float(n) / 10.0)+1)
        sus[0] = sus_ini
        exp[0] = exp_ini
        inf[0] = inf_ini
        rem[0] = rem_ini
    
        # Numerical solution: Runge-Kutta method
        i = 0
        for i in range(n):
            #if n <= 3170:
            b[i] = rt[i]
            k_sus_1 = f_sus(t_cou, sus_old, exp_old, inf_old, rem_old, b, a, r)
            k_exp_1 = f_exp(t_cou, sus_old, exp_old, inf_old, rem_old, b, a, r)
            k_inf_1 = f_inf(t_cou, sus_old, exp_old, inf_old, rem_old, b, a, r)
            k_rem_1 = f_rem(t_cou, sus_old, exp_old, inf_old, rem_old, b, a, r)
        
            k_sus_2 = f_sus(t_cou + 0.5 * dt, sus_old + k_sus_1 * 0.5 * dt, exp_old + k_exp_1 * 0.5 * dt, inf_old + k_inf_1 * 0.5 * dt, rem_old + k_rem_1 * 0.5 * dt, b, a, r)
            k_exp_2 = f_exp(t_cou + 0.5 * dt, sus_old + k_sus_1 * 0.5 * dt, exp_old + k_exp_1 * 0.5 * dt, inf_old + k_inf_1 * 0.5 * dt, rem_old + k_rem_1 * 0.5 * dt, b, a, r)
            k_inf_2 = f_inf(t_cou + 0.5 * dt, sus_old + k_sus_1 * 0.5 * dt, exp_old + k_exp_1 * 0.5 * dt, inf_old + k_inf_1 * 0.5 * dt, rem_old + k_rem_1 * 0.5 * dt, b, a, r)
            k_rem_2 = f_rem(t_cou + 0.5 * dt, sus_old + k_sus_1 * 0.5 * dt, exp_old + k_exp_1 * 0.5 * dt, inf_old + k_inf_1 * 0.5 * dt, rem_old + k_rem_1 * 0.5 * dt, b, a, r)
        
            k_sus_3 = f_sus(t_cou + 0.5 * dt, sus_old + k_sus_2 * 0.5 * dt, exp_old + k_exp_2 * 0.5 * dt, inf_old + k_inf_2 * 0.5 * dt, rem_old + k_rem_2 * 0.5 * dt, b, a, r)
            k_exp_3 = f_exp(t_cou + 0.5 * dt, sus_old + k_sus_2 * 0.5 * dt, exp_old + k_exp_2 * 0.5 * dt, inf_old + k_inf_2 * 0.5 * dt, rem_old + k_rem_2 * 0.5 * dt, b, a, r)
            k_inf_3 = f_inf(t_cou + 0.5 * dt, sus_old + k_sus_2 * 0.5 * dt, exp_old + k_exp_2 * 0.5 * dt, inf_old + k_inf_2 * 0.5 * dt, rem_old + k_rem_2 * 0.5 * dt, b, a, r)
            k_rem_3 = f_rem(t_cou + 0.5 * dt, sus_old + k_sus_2 * 0.5 * dt, exp_old + k_exp_2 * 0.5 * dt, inf_old + k_inf_2 * 0.5 * dt, rem_old + k_rem_2 * 0.5 * dt, b, a, r)
        
            k_sus_4 = f_sus(t_cou + dt, sus_old + k_sus_3 * dt, exp_old + k_exp_3 * dt, inf_old + k_inf_3 * dt, rem_old + k_rem_3 * dt, b, a, r)
            k_exp_4 = f_exp(t_cou + dt, sus_old + k_sus_3 * dt, exp_old + k_exp_3 * dt, inf_old + k_inf_3 * dt, rem_old + k_rem_3 * dt, b, a, r)
            k_inf_4 = f_inf(t_cou + dt, sus_old + k_sus_3 * dt, exp_old + k_exp_3 * dt, inf_old + k_inf_3 * dt, rem_old + k_rem_3 * dt, b, a, r)
            k_rem_4 = f_rem(t_cou + dt, sus_old + k_sus_3 * dt, exp_old + k_exp_3 * dt, inf_old + k_inf_3 * dt, rem_old + k_rem_3 * dt, b, a, r)
        
            sus_new = sus_old + dt / 6.0 * (k_sus_1 + 2.0 * k_sus_2 + 2.0 * k_sus_3 + k_sus_4)
            exp_new = exp_old + dt / 6.0 * (k_exp_1 + 2.0 * k_exp_2 + 2.0 * k_exp_3 + k_exp_4)
            inf_new = inf_old + dt / 6.0 * (k_inf_1 + 2.0 * k_inf_2 + 2.0 * k_inf_3 + k_inf_4)
            rem_new = rem_old + dt / 6.0 * (k_rem_1 + 2.0 * k_rem_2 + 2.0 * k_rem_3 + k_rem_4)
        
            sus_old = sus_new
            exp_old = exp_new
            inf_old = inf_new
            rem_old = rem_new
            
            sus[i+1] = sus_old
            exp[i+1] = exp_old
            inf[i+1] = inf_old
            rem[i+1] = rem_old 
        
        
            i += 1
            t_cou += dt
        
        k = 0
        for j in range(0, n+1):
            if j % int(1.0 / dt) == 0:
                sus_cal[k] = sus[j] #* n_total
                exp_cal[k] = exp[j] #* n_total
                inf_cal[k] = inf[j] #* n_total
                rem_cal[k] = rem[j] #* n_total
                inf_tot_cal[k] = (inf[j] + rem[j]) #* n_total
                k += 1                            
            
        inf_new_cal[0] = inf_tot_cal[0]
        NewI.iloc[0,t1] = inf_new_cal[0]
        for ij in range(1, int((float(n) / 10.0)+1.0)):
            inf_new_cal[ij] = inf_tot_cal[ij] - inf_tot_cal[ij - 1] 
            NewI.iloc[ij,t1] = inf_new_cal[ij]

    newI_dash = np.zeros(n_b+t_delay)
    for i_dash in range(n1+t_delay+1):
        newI_dash[i_dash]=NewI.iloc[i_dash,0]
    for i2_dash in range(n_b-n1-1):
        newI_dash[n1+t_delay+i2_dash+1]=NewI.iloc[n1+t_delay+i2_dash+1,i2_dash+1]

    b_file = 'NewI_all_' + str(t_delay) + '.csv'
    c_file = 'NewI_dash_' + str(t_delay) + '.csv'    
    NewI.to_csv(b_file)
    np.savetxt(c_file, newI_dash, delimiter=',', fmt='%.18e')
        
