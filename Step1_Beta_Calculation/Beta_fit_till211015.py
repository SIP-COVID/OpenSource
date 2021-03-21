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
from sklearn.metrics import r2_score
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
# Read csv
# Upload csv files
exdata_I_new = np.loadtxt("I_new.csv") # New infected peoplw
exdata_I_now = np.loadtxt("I_now.csv") # New infected peoplw
exdata_Rem = np.loadtxt("Rem1.csv") # New infected peoplw
# Initial conditions
a = 0.4
# Set gamma
r = 0.0691712976652005

n_total = 1.263 * 1.0e+8
exp_ini = 0.0 #110.0 / n_total
inf_ini = exdata_I_now[0] #0.142857143 #/ n_total
rem_ini = 0.0
sus_ini = n_total - exp_ini - inf_ini - rem_ini #1.0 #0.959
sus_old = sus_ini
exp_old = exp_ini
inf_old = inf_ini
rem_old = rem_ini

# Set the number of days
n = 3920
n1 = 392


rt = np.zeros(n)


beta_Data = np.zeros(n1)


E_calc = np.zeros(n1+1)
E_calc[0] = 0.0
b0 = np.zeros(n1)
b = np.zeros(n)
y = 0
x2 = 0
    

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
sus_old = sus_ini
exp_old = exp_ini
inf_old = inf_ini
rem_old = rem_ini

# Numerical solution: Runge-Kutta method
i = 0
for i in range(n):
    if i % int(1.0 / dt) == 0:
        E_calc[x2+1] = (exdata_I_new[x2+1]+exdata_I_new[x2+2])/(2*a)
        dE = E_calc[x2+1] - exp_old
        b0[x2] = (dE + a * exp_old)/(inf_old*sus_old)
        x2 += 1
    #if n <= 3170:
    b[i] = b0[x2-1]
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
for ij in range(1, int((float(n) / 10.0))):
    inf_new_cal[ij] = inf_tot_cal[ij] - inf_tot_cal[ij - 1]
 
beta_Data = b0 * n_total 
R2_I = r2_score(exdata_I_now[0:n1+1], inf_cal)
R2_I_new = r2_score(exdata_I_new[0:n1+1], inf_new_cal)
R2_R = r2_score(exdata_Rem[0:n1+1], rem_cal)



np.savetxt('beta_Data.csv', beta_Data, delimiter=',')
'''
# Output
m = int(float(n) / 10.0)
file = open('sus' + '.csv','w')
for l in range(0, m):
    file.writelines(str(sus_cal[l]))
    file.write('\n')
file.close()

file = open('exp' + '.csv','w')
for l in range(0, m):
    file.writelines(str(exp_cal[l]))
    file.write('\n')
file.close()

file = open('inf' + '.csv','w')
for l in range(0, m):
    file.writelines(str(inf_cal[l]))
    file.write('\n')
file.close()

file = open('inf_new' + '.csv','w')
for l in range(0, m):
    file.writelines(str(inf_new_cal[l]))
    file.write('\n')
file.close()

file = open('inf_tot' + '.csv','w')
for l in range(0, m):
    file.writelines(str(inf_tot_cal[l]))
    file.write('\n')
file.close()

file = open('rem' + '.csv','w')
for l in range(0, m):
    file.writelines(str(rem_cal[l]))
    file.write('\n')
file.close()
'''