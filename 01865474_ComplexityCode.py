# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:02:56 2023

@author: gerry
"""

#COMPLEXITY COURSEWORK
import numpy as np
import matplotlib.pyplot as plt
import random as r

def rand12(p):
    g = r.randint(1,100)
    p_abs = 100*p
    if g<p_abs:
        rand = 1
    else:
        rand = 2
    return rand

rand_list = []
for i in range(10):
    rand = rand12(0.5)
    rand_list.append(rand)


#%%
def init(L,p):#initialises an empty state of length L and random threshold slopes
    h_empty = np.zeros(L)
    z_th = []
    for i in range(L):
#        rand_z_th = r.randint(1,2)
#        z_th.append(rand_z_th)
        rand_z_th = rand12(p) 
        z_th.append(rand_z_th)
        
    return h_empty,np.array(z_th)

def drive(h):#adds 1 grain at 0 which is i=1
    h[0] += 1
    return h

def find_z(h):#converts an array of h into an array of z
    z=np.zeros(len(h))
    for i in range(len(h)):
        if i<(len(h)-1):
            z[i] = h[i] - h[i+1]
        else:
            z[i] = h[i]
    return z

def find_h(z):#converts an array of z into array of h
    z_rev = z[::-1]
    h_rev = np.zeros(len(z))
    for i in range(len(z_rev)):
        if i==0:
            h_rev[i] = z_rev[i]
        else:
            h_rev[i] = h_rev[i-1] + z_rev[i]
    h = h_rev[::-1]
    return h

def relax(z,z_th,p): #relaxation function
    s = 0
    while np.any(z > z_th):
        for i in range(len(z)):
            while z[i] > z_th[i]:
                if i == 0:
                    z[i] -= 2
                    z[i+1] += 1
                    s += 1
                    z_th[i] = rand12(p)
                elif i == (len(z)-1):
                    z[i] -= 1
                    z[i-1] += 1
                    s += 1
                    z_th[i] = rand12(p)
                else:
                    z[i] -= 2
                    z[i+1] += 1
                    z[i-1] += 1
                    s += 1
                    z_th[i] = rand12(p)
    return z,z_th,s

#if the pile hasn't reached a certain point across then no point checking for avalanches in this area

z_relaxed,z_th_new,s = relax([1,1,1,1,1,1],[1,1,1,1,1,1],0.5)
#%%
def oslo(L, n_grains, p):
    h, z_th = init(L,p) # gives empty h array and z_th
    s_list = []
    max_h_list = []
    for i in range(n_grains):#each iteration adds a grain and fully relaxes system
        h = drive(h) # adds a grain
        z = find_z(h) # converts the h array into a z array
        z, z_th, s = relax(z,z_th,p) # replaces the z array with the z array after relaxation and new z_th generated
        s_list.append(s) # adds the avalanche size, s, to the list
        h = find_h(z) # converts relaxed z array into relaxed h array 
        max_h_list.append(h[0]) # adds the maximum heihgt to a list
    return s_list, max_h_list, h
#%%
# Testing L=16 and L=32, p = 0.5

#L=16
s_L16, max_h_L16, h_L16 = oslo(16, 1000, 0.5)

x1 = np.linspace(0,999,1000)
plt.plot(x1,max_h_L16) # to find steady state
plt.show()
max_h_steady_L16 = max_h_L16[300:]
avrg_h_L16 = np.mean(max_h_steady_L16)
std_L16 = np.std(max_h_steady_L16,ddof=1)/np.sqrt(np.size(max_h_steady_L16))
print('Average max. h, after steady state, for L=16, p=0.5 =', avrg_h_L16,'+-',std_L16)

x11 = np.linspace(0,15,16)
plt.plot(x11,h_L16, '-')
plt.title('Steady State, L=16')
plt.xlabel('L')
plt.ylabel('h')
plt.show()

#L=32
s_L32, max_h_L32, h_L32 = oslo(32, 2000, 0.5)
x2 = np.linspace(0,1999,2000)
plt.plot(x2,max_h_L32)
plt.show()
max_h_steady_L32 = max_h_L32[1000:]
avrg_h_L32 = np.mean(max_h_steady_L32)
std_L32 = np.std(max_h_steady_L32,ddof=1)/np.sqrt(np.size(max_h_steady_L32))
print('Average max. h, after steady state, for L=32, p=0.5 =', avrg_h_L32,'+-',std_L32)

x22 = np.linspace(0,31,32)
plt.plot(x22,h_L32, '-')
plt.title('Steady State, L=32')
plt.xlabel('L')
plt.ylabel('h')
plt.show()
#%%
# Testing L=16 and L=32, p=0.1 --- should give max h close to 2L
#L=16
s_L16_p1, max_h_L16_p1, h_L16_p1 = oslo(16, 1000, 0.1)

x1 = np.linspace(0,999,1000)
plt.plot(x1,max_h_L16_p1) # to find steady state
plt.show()
max_h_steady_L16_p1 = max_h_L16_p1[300:]
avrg_h_L16_p1 = np.mean(max_h_steady_L16_p1)
print('Average max. h, after steady state, for L=16, p=0.5 =', avrg_h_L16_p1)

x11 = np.linspace(0,15,16)
plt.plot(x11,h_L16_p1, '-')
plt.title('Steady State, L=16, p=0.1')
plt.xlabel('L')
plt.ylabel('h')
plt.show()

#L=32
s_L32_p1, max_h_L32_p1, h_L32_p1 = oslo(32, 2000, 0.1)
x2 = np.linspace(0,1999,2000)
plt.plot(x2,max_h_L32_p1)
plt.show()
max_h_steady_L32_p1 = max_h_L32_p1[1250:]
avrg_h_L32_p1 = np.mean(max_h_steady_L32_p1)
print('Average max. h, after steady state, for L=32, p=0.5 =', avrg_h_L32_p1)

x22 = np.linspace(0,31,32)
plt.plot(x22,h_L32_p1, '-')
plt.title('Steady State, L=32, p=0.1')
plt.xlabel('L')
plt.ylabel('h')
plt.show()
#%%
#Testing different probabilities for L=32
s_L32,max_h_L32_1,h_L32_1 = oslo(32,2300,0)
s_L32,max_h_L32_2,h_L32_2 = oslo(32,2300,0.5)
s_L32,max_h_L32_3,h_L32_3 = oslo(32,2300,1)

max_h_L32_1_ss = max_h_L32_1[1300:]
max_h_L32_2_ss = max_h_L32_2[1300:]
max_h_L32_3_ss = max_h_L32_3[1300:]

x_test = np.linspace(0,2300,2300)
plt.plot(x_test, max_h_L32_1, label='$p_1=0, p_2=1$')
plt.plot(x_test, max_h_L32_2, label='$p_1=0.5, p_2=0.5$')
plt.plot(x_test, max_h_L32_3, label='$p_1=1, p_2=0$')
plt.legend()
plt.grid()
plt.xlabel('Time (No. grains added)')
plt.ylabel('Maximum height')
plt.show()

x_test_2 = np.linspace(0,32,32)
plt.plot(x_test_2,h_L32_1, label='$p_1=0, p_2=1$')
plt.plot(x_test_2,h_L32_2, label='$p_1=0.5, p_2=0.5$')
plt.plot(x_test_2,h_L32_3, label='$p_1=1, p_2=0$')
plt.legend()
plt.grid()
plt.xlabel('Site i')
plt.ylabel('Height')
plt.show()
#%%
#Task 2a
#plot max h as a function of t (or grains added)

#L=4
s_4, max_h_4, h_4 = oslo(4,5000,0.5) 
plt.plot(max_h_4, label='L=4')
#L=8
s_8, max_h_8, h_8 = oslo(8,5000,0.5)
plt.plot(max_h_8, label='L=8')
#L=16
s_16,max_h_16,h_16 = oslo(16,5000,0.5)
plt.plot(max_h_16, label='L=16')
#L=32
s_32,max_h_32,h_32 = oslo(32,5000,0.5)
plt.plot(max_h_32,label='L=32')
#L=64
s_64,max_h_64,h_64 = oslo(64,10000,0.5)
plt.plot(max_h_64, label='L=64')
#L=128
s_128,max_h_128,h_128 = oslo(128,21000,0.5)
plt.plot(max_h_128, label='L=128')
#L=256
s_256, max_h_256,h_256 = oslo(256, 65000,0.5)
plt.plot(max_h_256, label='L=256')

plt.ylabel('Pile Height')
plt.xlabel('Time (No. grains added)')
plt.title('Pile height as a function of time')
plt.legend()
plt.grid()
plt.show()

#%%
#Task 2b
#finding average crossover time (time for a grain to leave the system)
#need to remake oslo algo for different outputs

def relax_2b(z,z_th,p): #relaxation function
    s = 0
    crossover = 0
    while np.any(z > z_th):
        for i in range(len(z)):
            while z[i] > z_th[i]:
                if i == 0:
                    z[i] -= 2
                    z[i+1] += 1
                    s += 1
                    z_th[i] = rand12(p)
                elif i == (len(z)-1):
                    z[i] -= 1
                    z[i-1] += 1
                    s += 1
                    z_th[i] = rand12(p)
                    crossover += 1
                else:
                    z[i] -= 2
                    z[i+1] += 1
                    z[i-1] += 1
                    s += 1
                    z_th[i] = rand12(p)
    return z,z_th,s,crossover

def oslo_2b(L, n_grains, p):
    h, z_th = init(L,p) # gives empty h array and z_th
    s_list = []
    max_h_list = []
    crossover_list = []
    for i in range(n_grains):#each iteration adds a grain and fully relaxes system
        h = drive(h) # adds a grain
        z = find_z(h) # converts the h array into a z array
        z, z_th, s, crossover = relax_2b(z,z_th,p) # replaces the z array with the z array after relaxation and new z_th generated
        s_list.append(s) # adds the avalanche size, s, to the list
        h = find_h(z) # converts relaxed z array into relaxed h array 
        max_h_list.append(h[0]) # adds the maximum heihgt to a list
        crossover_list.append(crossover)
    return s_list, max_h_list, h, crossover_list

def crossover_finder(crossover_list):# finds the index of the first non-zero element of the crossover list
    crossover_time = crossover_list.index(next(filter(lambda x: x!=0, crossover_list)))
    return crossover_time+1 # +1 to make a real time

#cross_time_test = crossover_finder([0,0,0,0,1,0,1])
#%%
#L=32
s_32,max_h_32,h_32,crossover_32 = oslo_2b(32,2000,0.5)
crossover_time_32 = crossover_finder(crossover_32) #crossover time for one test with L=32

#Finding average crossover time for the range of Ls

cross_4 = []
for i in range(0,10):
    s_4,max_h_4,h_4,crossover_4 = oslo_2b(4,1000,0.5)
    crossover_time_4 = crossover_finder(crossover_4)
    cross_4.append(crossover_time_4)
av_cross_4 = np.mean(cross_4)

cross_8 = []
for i in range(0,10):
    s_8,max_h_8,h_8,crossover_8 = oslo_2b(8,1000,0.5)
    crossover_time_8 = crossover_finder(crossover_8)
    cross_8.append(crossover_time_8)
av_cross_8 = np.mean(cross_8)

cross_16 = []
for i in range(0,10):
    s_16,max_h_16,h_16,crossover_16 = oslo_2b(16,1000,0.5)
    crossover_time_16 = crossover_finder(crossover_16)
    cross_16.append(crossover_time_16)
av_cross_16 = np.mean(cross_16)

cross_32 = []
for i in range(0,10):
    s_32,max_h_32,h_32,crossover_32 = oslo_2b(32,2000,0.5)
    crossover_time_32 = crossover_finder(crossover_32)
    cross_32.append(crossover_time_32)
av_cross_32 = np.mean(cross_32)
#%%
cross_64 = []
for i in range(0,10):
    s_64,max_h_64,h_64,crossover_64 = oslo_2b(64,5000,0.5)
    crossover_time_64 = crossover_finder(crossover_64)
    cross_64.append(crossover_time_64)
av_cross_64 = np.mean(cross_64)
#%%
cross_128 = []
for i in range(0,10):
    s_128,max_h_128,h_128,crossover_128 = oslo_2b(128,16000,0.5)
    crossover_time_128 = crossover_finder(crossover_128)
    cross_128.append(crossover_time_128)
av_cross_128 = np.mean(cross_128)
#%% SLOW TO RUN
cross_256 = []
for i in range(0,10):
    s_256,max_h_256,h_256,crossover_256 = oslo_2b(256,59000,0.5)
    crossover_time_256 = crossover_finder(crossover_256)
    cross_256.append(crossover_time_256)
av_cross_256 = np.mean(cross_256)
#%%
#Plot for 2b

av_cross_list = []
av_cross_list.append(av_cross_4)
av_cross_list.append(av_cross_8)
av_cross_list.append(av_cross_16)
av_cross_list.append(av_cross_32)
av_cross_list.append(av_cross_64)
av_cross_list.append(av_cross_128)
av_cross_list.append(av_cross_256)
L_list = [4,8,16,32,64,128,256]

plt.plot(L_list, av_cross_list, 'x', color = 'black')
plt.title('Average cross-over time as a function of system size')
plt.xlabel('L')
plt.ylabel('Average cross-over time')
plt.show()
#%%
import scipy.optimize as sci
def quad(x,a,b,c):
    return a*x**2 + b*x + c
popt,v = sci.curve_fit(quad,L_list,av_cross_list)
a,b,c = popt
err = v[1,1]**0.5
x = np.linspace(0,260,500)    
plt.plot(x,quad(x,a,b,c), '--',color='black')
plt.plot(L_list, av_cross_list, 'x', color = 'black')
plt.title('Average cross-over time as a function of system size')
plt.xlabel('L')
plt.ylabel('Average cross-over time')
plt.show()

plt.plot(x,quad(x,a,b,c), '--',color='black')
plt.plot(L_list, av_cross_list, 'x', color = 'black')
#plt.title('Average cross-over time as a function of system size')
plt.xlabel('L')
plt.ylabel('Average cross-over time')
plt.xlim(0,17)
plt.ylim(0,250)
plt.show()

plt.plot(x,quad(x,a,b,c), '--',color='black')
plt.plot(L_list, av_cross_list, 'x', color = 'black')
#plt.title('Average cross-over time as a function of system size')
plt.xlabel('L')
plt.ylabel('Average cross-over time')
plt.xlim(250,258)
plt.ylim(52000,58000)
plt.show()
#%%
#TASK 2d - Processed height and data collapse
#L=4
max_h_4_list = []
for i in range(0,5):
    s_4,max_h_4,h_4 = oslo(4,1000,0.5)
    max_h_4_list.append(max_h_4)

max_h_4_list = np.array(max_h_4_list)

processed_h_4 = (max_h_4_list[0] + max_h_4_list[1] + max_h_4_list[2] + max_h_4_list[3] + max_h_4_list[4])/5

#L=8
max_h_8_list = []
for i in range(0,5):
    s_8,max_h_8,h_8 = oslo(8,1000,0.5)
    max_h_8_list.append(max_h_8)

max_h_8_list = np.array(max_h_8_list)

processed_h_8 = (max_h_8_list[0] + max_h_8_list[1] + max_h_8_list[2] + max_h_8_list[3] + max_h_8_list[4])/5

#L=16
max_h_16_list = []
for i in range(0,5):
    s_16,max_h_16,h_16 = oslo(16,1000,0.5)
    max_h_16_list.append(max_h_16)

max_h_16_list = np.array(max_h_16_list)

processed_h_16 = (max_h_16_list[0] + max_h_16_list[1] + max_h_16_list[2] + max_h_16_list[3] + max_h_16_list[4])/5

#L=32
max_h_32_list = []
for i in range(0,5):
    s_32,max_h_32,h_32 = oslo(32,2000,0.5)
    max_h_32_list.append(max_h_32)

max_h_32_list = np.array(max_h_32_list)

processed_h_32 = (max_h_32_list[0] + max_h_32_list[1] + max_h_32_list[2] + max_h_32_list[3] + max_h_32_list[4])/5
#%%
#L=64
max_h_64_list = []
for i in range(0,5):
    s_64,max_h_64,h_64 = oslo(64,5000,0.5)
    max_h_64_list.append(max_h_64)

max_h_64_list = np.array(max_h_64_list)

processed_h_64 = (max_h_64_list[0] + max_h_64_list[1] + max_h_64_list[2] + max_h_64_list[3] + max_h_64_list[4])/5
#%%
#L=128
max_h_128_list = []
for i in range(0,5):
    s_128,max_h_128,h_128 = oslo(128,16000,0.5)
    max_h_128_list.append(max_h_128)

max_h_128_list = np.array(max_h_128_list)

processed_h_128 = (max_h_128_list[0] + max_h_128_list[1] + max_h_128_list[2] + max_h_128_list[3] + max_h_128_list[4])/5
#%%
#L=256
max_h_256_list = []
for i in range(0,5):
    s_256,max_h_256,h_256 = oslo(256,59000,0.5)
    max_h_256_list.append(max_h_256)

max_h_256_list = np.array(max_h_256_list)

processed_h_256 = (max_h_256_list[0] + max_h_256_list[1] + max_h_256_list[2] + max_h_256_list[3] + max_h_256_list[4])/5
#%%
plt.plot(processed_h_4, label='L=4')
plt.plot(processed_h_8, label='L=8')
plt.plot(processed_h_16, label='L=16')
plt.plot(processed_h_32, label='L=32')
plt.plot(processed_h_64, label='L=64')
plt.plot(processed_h_128, label='L=128')
plt.plot(processed_h_256, label='L=256')
plt.title('Processed height as a function of time')
plt.xlabel('Time (No. grains added)')
plt.ylabel('Processed height')
plt.legend()
plt.grid()
plt.show()
#%%
scaled_h_4 = processed_h_4/4
scaled_h_8 = processed_h_8/8
scaled_h_16 = processed_h_16/16
scaled_h_32 = processed_h_32/32
scaled_h_64 = processed_h_64/64
scaled_h_128 = processed_h_128/128
scaled_h_256 = processed_h_256/256

t_4 = np.linspace(0,999,1000)
scaled_t_4 = t_4/4**2

scaled_t_8 = t_4/8**2
scaled_t_16 = t_4/16**2

t_32 = np.linspace(0,1999,2000)
scaled_t_32 = t_32/32**2

t_64 = np.linspace(0,4999,5000)
scaled_t_64 = t_64/64**2

t_128 = np.linspace(0,15999,16000)
scaled_t_128 = t_128/128**2

t_256 = np.linspace(0,58999,59000)
scaled_t_256 = t_256/256**2

plt.plot(scaled_t_4,scaled_h_4,label='L=4')
plt.plot(scaled_t_8,scaled_h_8,label='L=8')
plt.plot(scaled_t_16,scaled_h_16,label='L=16')
plt.plot(scaled_t_32,scaled_h_32,label='L=32')
plt.plot(scaled_t_64,scaled_h_64,label='L=64')
plt.plot(scaled_t_128,scaled_h_128,label='L=128')
plt.plot(scaled_t_256,scaled_h_256,label='L=256')
plt.xlim(-0.1, 1.2)
plt.legend()
plt.xlabel('$t / L^2$')
plt.ylabel('h / L')
plt.title('Data collapse for the processed height as a function of time')
plt.show()

#%%
#TASK 2e
T = 10000
#L=4
s_4,max_h_4,h_4 = oslo(4,10030,0.5)
max_h_4_ss = max_h_4[30:]
av_h_4_ss = sum(max_h_4_ss)/T

#L=8
s_8,max_h_8,h_8 = oslo(8,10060,0.5)
max_h_8_ss = max_h_8[60:]
av_h_8_ss = sum(max_h_8_ss)/T

#L=16
s_16,max_h_16,h_16 = oslo(16,10300,0.5)
max_h_16_ss = max_h_16[300:]
av_h_16_ss = sum(max_h_16_ss)/T

#L=32
s_32,max_h_32,h_32 = oslo(32,11300,0.5)
max_h_32_ss = max_h_32[1300:]
av_h_32_ss = sum(max_h_32_ss)/T

#L=64
s_64,max_h_64,h_64 = oslo(64,15000,0.5)
max_h_64_ss = max_h_64[5000:]
av_h_64_ss = sum(max_h_64_ss)/T

#L=128
s_128,max_h_128,h_128 = oslo(128,25000,0.5)
max_h_128_ss = max_h_128[15000:]
av_h_128_ss = sum(max_h_128_ss)/T
#%%
#L=256
s_256,max_h_256,h_256 = oslo(256,70000,0.5)
max_h_256_ss = max_h_256[60000:]
av_h_256_ss = sum(max_h_256_ss)/T
#%%
L_list = [4,8,16,32,64,128,256]
av_h_list = [av_h_4_ss,av_h_8_ss,av_h_16_ss,av_h_32_ss,av_h_64_ss,av_h_128_ss,av_h_256_ss]
plt.plot(L_list,av_h_list,'.')
plt.show()

#fitting to find a0 w1 and a1

import scipy.optimize as sci

def mean_h_func(L,a0,a1,w1):
    return a0*L - a0*a1*L**(1-w1)

popt,v = sci.curve_fit(mean_h_func,L_list,av_h_list)
a0,a1,w1 = popt
err_a0 = np.sqrt(v[0,0])
err_a1 = np.sqrt(v[1,1])
err_w1 = np.sqrt(v[2,2])

x = np.linspace(0,260,500)
plt.plot(x,mean_h_func(x,a0,a1,w1), '--',color='black')
plt.plot(L_list, av_h_list, 'x', color = 'black')
plt.xlabel('L')
plt.ylabel('Mean recurring configuration height')
plt.title('Mean recurring configuration height as a function of system size')
plt.show()

print('a0 = ',a0,'+-',err_a0, 'a1 = ',a1,'+-',err_a1, 'w1 = ',w1,'+-',err_w1)
#%%
#2e re done
def a0_func(L,a0):
    return a0*L
x = np.linspace(0,260,500)
L_list_a0 = [64,128,256]
av_h_a0 = [av_h_64_ss,av_h_128_ss,av_h_256_ss]
popt,v = sci.curve_fit(a0_func,L_list_a0,av_h_a0)
a0 = popt
err_a0 = np.sqrt(v[0,0])
plt.plot(x,a0_func(x,a0),'--',color='black')
plt.plot(L_list, av_h_list, 'x', color = 'black')
plt.xlabel('L')
plt.ylabel('Mean recurring configuration height')
plt.show()
print(a0, err_a0)
#%%

def w1_func(log_L,w1,log_c):
    return -w1*log_L + log_c
L_list_w1 = [4,8,16,32]
av_h_w1 = [av_h_4_ss,av_h_8_ss,av_h_16_ss,av_h_32_ss]
y = 1 - av_h_w1/(a0*L_list_w1)
y2 = 1 - av_h_list/(a0*L_list)
popt,v = sci.curve_fit(w1_func,np.log10(L_list_w1),np.log10(y))
w1,log_c = popt
err_w1 = np.sqrt(v[0,0])

x=np.linspace(1,260,500)
c = 10**log_c
y3 = c*x**(-w1)

plt.plot(x,y3,'--',color='blue')
plt.plot(L_list, y2, 'x', color='blue')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('L')
plt.ylabel(r'$1 - \langle h \rangle / a_0L$')
plt.show()
print(w1, err_w1)
#%%
#TASK 2f
T=10000
#already have the average heights for each L
#need the average of the heights squared 
#L=4
max_h_4_ss = np.array(max_h_4[30:])
max_h_4_ss_sq = max_h_4_ss**2
av_h_4_ss_sq = sum(max_h_4_ss_sq)/T

sig_4 = np.sqrt(av_h_4_ss_sq - av_h_4_ss**2)

#L=8
max_h_8_ss = np.array(max_h_8[60:])
max_h_8_ss_sq = max_h_8_ss**2
av_h_8_ss_sq = sum(max_h_8_ss_sq)/T

sig_8 = np.sqrt(av_h_8_ss_sq - av_h_8_ss**2)

#L=16
max_h_16_ss = np.array(max_h_16[300:])
max_h_16_ss_sq = max_h_16_ss**2
av_h_16_ss_sq = sum(max_h_16_ss_sq)/T

sig_16 = np.sqrt(av_h_16_ss_sq - av_h_16_ss**2)

#L=32
max_h_32_ss = np.array(max_h_32[1300:])
max_h_32_ss_sq = max_h_32_ss**2
av_h_32_ss_sq = sum(max_h_32_ss_sq)/T

sig_32 = np.sqrt(av_h_32_ss_sq - av_h_32_ss**2)

#L=64
max_h_64_ss = np.array(max_h_64[5000:])
max_h_64_ss_sq = max_h_64_ss**2
av_h_64_ss_sq = sum(max_h_64_ss_sq)/T

sig_64 = np.sqrt(av_h_64_ss_sq - av_h_64_ss**2)

#L=128
max_h_128_ss = np.array(max_h_128[15000:])
max_h_128_ss_sq = max_h_128_ss**2
av_h_128_ss_sq = sum(max_h_128_ss_sq)/T

sig_128 = np.sqrt(av_h_128_ss_sq - av_h_128_ss**2)

#L=256
max_h_256_ss = np.array(max_h_256[60000:])
max_h_256_ss_sq = max_h_256_ss**2
av_h_256_ss_sq = sum(max_h_256_ss_sq)/T

sig_256 = np.sqrt(av_h_256_ss_sq - av_h_256_ss**2)

sig_list = [sig_4,sig_8,sig_16,sig_32,sig_64,sig_128,sig_256]

plt.plot(L_list,sig_list,'x',color='black')
plt.xlabel('L')
plt.ylabel('Standard deviation of the height')
plt.title('Standard deviation of the mean recurring configuration height as a function of system size')
plt.grid()
plt.show()

def sqrt(L,a,b):
    return a*np.sqrt(L) + b

popt,v = sci.curve_fit(sqrt,L_list,sig_list)
a,b = popt
#err = v[1,1]**0.5
x = np.linspace(0,260,500)
plt.plot(x,sqrt(x,a,b), '--',color='black')
plt.plot(L_list, sig_list, 'x', color = 'black')
plt.xlabel('L')
plt.ylabel('Standard deviation of the height')
#plt.title('Standard deviation of the height as a function of system size')
plt.grid()
plt.show()
#%%
#2f re-done
def sig_fit(L,m,c):
    sig = L**m + c
    return sig

popt,v = sci.curve_fit(sig_fit,(L_list),sig_list)
m,c = popt
err_m = v[0,0]
x = np.linspace(1,270,500)
y = x**m + 10**c
plt.plot(x,sig_fit(x,m,c),'--',color='black')
plt.plot(L_list, sig_list,'x',color='black')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('L')
plt.ylabel('Standard deviation of the height')
plt.xlim(3,270)
plt.grid()
plt.show()
print(m, err_m)
#%%
#TASK 2g
#count the number of each height for each L and divide by 10,000
N = 10000
max_h_4_ss = max_h_4_ss.tolist()
min_4 = int(min(max_h_4_ss))
max_4 = int(max(max_h_4_ss))
poss_4 = [i for i in range(min_4,max_4+1)]
freq_4 = []
for i in poss_4:
    freq_4.append(max_h_4_ss.count(i)) 
Ph_4 = np.array(freq_4)/N

max_h_8_ss = max_h_8_ss.tolist()
min_8 = int(min(max_h_8_ss))
max_8 = int(max(max_h_8_ss))
poss_8 = [i for i in range(min_8,max_8+1)]
freq_8 = []
for i in poss_8:
    freq_8.append(max_h_8_ss.count(i))
Ph_8 = np.array(freq_8)/N

max_h_16_ss = max_h_16_ss.tolist()
min_16 = int(min(max_h_16_ss))
max_16 = int(max(max_h_16_ss))
poss_16 = [i for i in range(min_16,max_16+1)]
freq_16 = []
for i in poss_16:
    freq_16.append(max_h_16_ss.count(i))
Ph_16 = np.array(freq_16)/N

max_h_32_ss = max_h_32_ss.tolist()
min_32 = int(min(max_h_32_ss))
max_32 = int(max(max_h_32_ss))
poss_32 = [i for i in range(min_32,max_32+1)]
freq_32 = []
for i in poss_32:
    freq_32.append(max_h_32_ss.count(i))
Ph_32 = np.array(freq_32)/N

max_h_64_ss = max_h_64_ss.tolist()
min_64 = int(min(max_h_64_ss))
max_64 = int(max(max_h_64_ss))
poss_64 = [i for i in range(min_64,max_64+1)]
freq_64 = []
for i in poss_64:
    freq_64.append(max_h_64_ss.count(i))
Ph_64 = np.array(freq_64)/N

max_h_128_ss = max_h_128_ss.tolist()
min_128 = int(min(max_h_128_ss))
max_128 = int(max(max_h_128_ss))
poss_128 = [i for i in range(min_128,max_128+1)]
freq_128 = []
for i in poss_128:
    freq_128.append(max_h_128_ss.count(i))
Ph_128 = np.array(freq_128)/N

max_h_256_ss = max_h_256_ss.tolist()
min_256 = int(min(max_h_256_ss))
max_256 = int(max(max_h_256_ss))
poss_256 = [i for i in range(min_256,max_256+1)]
freq_256 = []
for i in poss_256:
    freq_256.append(max_h_256_ss.count(i))
Ph_256 = np.array(freq_256)/N
#%%
plt.plot(poss_4,Ph_4,label='L=4')
plt.plot(poss_8,Ph_8,label='L=8')
plt.plot(poss_16,Ph_16,label='L=16')
plt.plot(poss_32,Ph_32,label='L=32')
plt.plot(poss_64,Ph_64,label='L=64')
plt.plot(poss_128,Ph_128,label='L=128')
plt.plot(poss_256,Ph_256,label='L=256')
plt.legend()
plt.grid()
#plt.title('Height Probability') # over 10,000 steady state heights
plt.xlabel('h')
plt.ylabel('P(h;L)')
plt.show()
#%%
#PDF*sigma vs mean_h/sigma
#L=4
Ph_sig_4 = Ph_4*sig_4
h_sig_4 = (np.array(poss_4)-av_h_4_ss)/sig_4

Ph_sig_8 = Ph_8*sig_8
h_sig_8 = (np.array(poss_8)-av_h_8_ss)/sig_8

Ph_sig_16 = Ph_16*sig_16
h_sig_16 = (np.array(poss_16)-av_h_16_ss)/sig_16

Ph_sig_32 = Ph_32*sig_32
h_sig_32 = (np.array(poss_32)-av_h_32_ss)/sig_32

Ph_sig_64 = Ph_64*sig_64
h_sig_64 = (np.array(poss_64)-av_h_64_ss)/sig_64

Ph_sig_128 = Ph_128*sig_128
h_sig_128 = (np.array(poss_128)-av_h_128_ss)/sig_128

Ph_sig_256 = Ph_256*sig_256
h_sig_256 = (np.array(poss_256)-av_h_256_ss)/sig_256

plt.plot(h_sig_4,Ph_sig_4,label='L=4')
plt.plot(h_sig_8,Ph_sig_8,label='L=8')
plt.plot(h_sig_16,Ph_sig_16,label='L=16')
plt.plot(h_sig_32,Ph_sig_32,label='L=32')
plt.plot(h_sig_64,Ph_sig_64,label='L=64')
plt.plot(h_sig_128,Ph_sig_128,label='L=128')
plt.plot(h_sig_256,Ph_sig_256,label='L=256')
plt.legend()
plt.grid()
#plt.title('Data Collapse for P(h;L)')
plt.xlabel('$(h-<h>)/ \sigma$ ')
plt.ylabel('$\sigma *P(h;L)$')
plt.show()

#%%
#Task 2g iii)
def gauss(h,mu,sig):
    gaussian = (1/(sig*np.sqrt(2*np.pi)))*np.exp(-((h-mu)**2)/(2*sig**2))
    return gaussian

h_sig_list = h_sig_4.tolist()+h_sig_8.tolist()+h_sig_16.tolist()+h_sig_32.tolist()+h_sig_64.tolist()+h_sig_128.tolist()+h_sig_256.tolist()
Ph_sig_list = Ph_sig_4.tolist()+Ph_sig_8.tolist()+Ph_sig_16.tolist()+Ph_sig_32.tolist()+Ph_sig_64.tolist()+Ph_sig_128.tolist()+Ph_sig_256.tolist()

#%%
#fitting a gaussian to the collapsed data --- Instead should compare the guassian in notes to the data
popt,v = sci.curve_fit(gauss,h_sig_list,Ph_sig_list)
mu,sig = popt
err_mu = np.sqrt(v[0][0])
err_sig = np.sqrt(v[1][1])
h = np.linspace(-5,7,500)
#plt.plot(h_sig_list, Ph_sig_list, 'x', color = 'black')
plt.plot(h_sig_4,Ph_sig_4,label='L=4')
plt.plot(h_sig_8,Ph_sig_8,label='L=8')
plt.plot(h_sig_16,Ph_sig_16,label='L=16')
plt.plot(h_sig_32,Ph_sig_32,label='L=32')
plt.plot(h_sig_64,Ph_sig_64,label='L=64')
plt.plot(h_sig_128,Ph_sig_128,label='L=128')
plt.plot(h_sig_256,Ph_sig_256,label='L=256')
plt.plot(h,gauss(h,mu,sig), '--',color='black',label='fit')
plt.legend()
plt.xlabel('$(h-<h>)/ \sigma$')
plt.ylabel('$\sigma *P(h;L)$')
plt.title('Data collapse for P(h;L)')
plt.grid()
plt.show()
print(mu, '+-', err_mu, sig, '+-', err_sig)
#%%
#finding how many data points are within 1 std dev from the mean 
lower_bound = mu - sig
upper_bound = mu + sig
outside_1_std_dev = []
for i in h_sig_list:
    if i<lower_bound:
        outside_1_std_dev.append(i)
    if i>upper_bound:
        outside_1_std_dev.append(i)
ratio_inside_1_std_dev = 1 - (len(outside_1_std_dev)/len(h_sig_list))
print('% of observations within 1 std dev of mean =',ratio_inside_1_std_dev)
#finding how many data points are within 2 std dev from the mean 
lower_bound_2 = mu - 2*sig
upper_bound_2 = mu + 2*sig
outside_2_std_dev = []
for i in h_sig_list:
    if i<lower_bound_2:
        outside_2_std_dev.append(i)
    if i>upper_bound_2:
        outside_2_std_dev.append(i)
ratio_inside_2_std_dev = 1 - (len(outside_2_std_dev)/len(h_sig_list))
print('% of observations within 2 std dev of mean =',ratio_inside_2_std_dev)
#%%
#comparing the expected gaussian with the data
def exp_gauss(h):
    exp_gauss = (1/np.sqrt(2*np.pi))*np.exp(-0.5*h**2)
    return exp_gauss

exp_Ph_sig = exp_gauss(h)

plt.plot(h_sig_4,Ph_sig_4,label='L=4')
plt.plot(h_sig_8,Ph_sig_8,label='L=8')
plt.plot(h_sig_16,Ph_sig_16,label='L=16')
plt.plot(h_sig_32,Ph_sig_32,label='L=32')
plt.plot(h_sig_64,Ph_sig_64,label='L=64')
plt.plot(h_sig_128,Ph_sig_128,label='L=128')
plt.plot(h_sig_256,Ph_sig_256,label='L=256')
plt.plot(h,exp_Ph_sig, '--',color='black', label='Theoretical prediction')
plt.legend()
plt.xlabel('$(h-<h>)/ \sigma$')
plt.ylabel('$\sigma *P(h;L)$')
#plt.title('Data collapse for P(h;L)')
plt.xlim(-4,5)
plt.grid()
plt.show()

#%%
#Task 3a)i) - logbinning

def logbin(data, scale = 1., zeros = False):
    """
    logbin(data, scale = 1., zeros = False)

    Log-bin frequency of unique integer values in data. Returns probabilities
    for each bin.

    Array, data, is a 1-d array containing full set of event sizes for a
    given process in no particular order. For instance, in the Oslo Model
    the array may contain the avalanche size recorded at each time step. For
    a complex network, the array may contain the degree of each node in the
    network. The logbin function finds the frequency of each unique value in
    the data array. The function then bins these frequencies in logarithmically
    increasing bin sizes controlled by the scale parameter.

    Minimum binsize is always 1. Bin edges are lowered to nearest integer. Bins
    are always unique, i.e. two different float bin edges corresponding to the
    same integer interval will not be included twice. Note, rounding to integer
    values results in noise at small event sizes.

    Parameters
    ----------

    data: array_like, 1 dimensional, non-negative integers
          Input array. (e.g. Raw avalanche size data in Oslo model.)

    scale: float, greater or equal to 1.
          Scale parameter controlling the growth of bin sizes.
          If scale = 1., function will return frequency of each unique integer
          value in data with no binning.

    zeros: boolean
          Set zeros = True if you want binning function to consider events of
          size 0.
          Note that output cannot be plotted on log-log scale if data contains
          zeros. If zeros = False, events of size 0 will be removed from data.

    Returns
    -------

    x: array_like, 1 dimensional
          Array of coordinates for bin centres calculated using geometric mean
          of bin edges. Bins with a count of 0 will not be returned.
    y: array_like, 1 dimensional
          Array of normalised frequency counts within each bin. Bins with a
          count of 0 will not be returned.
    """
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    smax = np.max(data)
    if scale > 1:
        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)
        count = count.astype('float')
        for i in range(len(y)):
            y[i] = np.sum(count[binedges[i]:binedges[i+1]]/(binedges[i+1] - binedges[i]))
            # print(binedges[i],binedges[i+1])
        # print(smax,jmax,binedges,x)
        # print(x,y)
    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
    y /= tot
    x = x[y!=0]
    y = y[y!=0]
    return x,y
#%%
#Saving the s lists because i am using N=1,000,000 - too long to run each time
s_4, max_h_4_big, h_4_big = oslo(4,1000030,0.5)
#%%
np.savetxt('s_4.csv',s_4,delimiter=',')
#%%
s_8, max_h_8_big, h_8_big = oslo(8,1000060,0.5)
#%%
np.savetxt('s_8.csv',s_8,delimiter=',')
#%%
s_16, max_h_16_big, h_16_big = oslo(16,1000300,0.5)
np.savetxt('s_16.csv',s_16,delimiter=',')

s_32, max_h_32_big, h_32_big = oslo(32,1001300,0.5)
np.savetxt('s_32.csv',s_32,delimiter=',')

s_64, max_h_64_big, h_64_big = oslo(64,1005000,0.5)
np.savetxt('s_64.csv',s_64,delimiter=',')

s_128, max_h_128_big, h_128_big = oslo(128,1015000,0.5)
np.savetxt('s_128.csv',s_128,delimiter=',')

s_256, max_h_256_big, h_256_big = oslo(256,1060000,0.5)
np.savetxt('s_256.csv',s_256,delimiter=',')
#%%
#Cut the steady state part out of the s lists
s_4_ss = s_4[30:]
s_8_ss = s_8[60:]
s_16_ss = s_16[300:]
s_32_ss = s_32[1300:]
s_64_ss = s_64[5000:]
s_128_ss = s_128[15000:]
s_256_ss = s_256[60000:]
#%%
#Binning
bins_4,bin_freq_4 = logbin(s_4_ss,scale=1,zeros=False)
bins_8,bin_freq_8 = logbin(s_8_ss,scale=1,zeros=False)
bins_16,bin_freq_16 = logbin(s_16_ss,scale=1,zeros=False)
bins_32,bin_freq_32 = logbin(s_32_ss,scale=1,zeros=False)
bins_64,bin_freq_64 = logbin(s_64_ss,scale=1,zeros=False)
bins_128,bin_freq_128 = logbin(s_128_ss,scale=1,zeros=False)
bins_256,bin_freq_256 = logbin(s_256_ss,scale=1,zeros=False)

plt.plot(bins_4,bin_freq_4,label='L=4')
plt.plot(bins_8,bin_freq_8,label='L=8')
plt.plot(bins_16,bin_freq_16,label='L=16')
plt.plot(bins_32,bin_freq_32,label='L=32')
plt.plot(bins_64,bin_freq_64,label='L=64')
plt.plot(bins_128,bin_freq_128,label='L=128')
plt.plot(bins_256,bin_freq_256,label='L=256')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
#%%
bins_4,bin_freq_4 = logbin(s_4_ss,scale=2,zeros=False)
bins_8,bin_freq_8 = logbin(s_8_ss,scale=2,zeros=False)
bins_16,bin_freq_16 = logbin(s_16_ss,scale=2,zeros=False)
bins_32,bin_freq_32 = logbin(s_32_ss,scale=2,zeros=False)
bins_64,bin_freq_64 = logbin(s_64_ss,scale=2,zeros=False)
bins_128,bin_freq_128 = logbin(s_128_ss,scale=2,zeros=False)
bins_256,bin_freq_256 = logbin(s_256_ss,scale=2,zeros=False)

plt.plot(bins_4,bin_freq_4,label='L=4')
plt.plot(bins_8,bin_freq_8,label='L=8')
plt.plot(bins_16,bin_freq_16,label='L=16')
plt.plot(bins_32,bin_freq_32,label='L=32')
plt.plot(bins_64,bin_freq_64,label='L=64')
plt.plot(bins_128,bin_freq_128,label='L=128')
plt.plot(bins_256,bin_freq_256,label='L=256')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
#%%
#THIS ONE
bins_4,bin_freq_4 = logbin(s_4_ss,scale=1.5,zeros=False)
bins_8,bin_freq_8 = logbin(s_8_ss,scale=1.5,zeros=False)
bins_16,bin_freq_16 = logbin(s_16_ss,scale=1.5,zeros=False)
bins_32,bin_freq_32 = logbin(s_32_ss,scale=1.5,zeros=False)
bins_64,bin_freq_64 = logbin(s_64_ss,scale=1.5,zeros=False)
bins_128,bin_freq_128 = logbin(s_128_ss,scale=1.5,zeros=False)
bins_256,bin_freq_256 = logbin(s_256_ss,scale=1.5,zeros=False)

plt.plot(bins_4,bin_freq_4,label='L=4')
plt.plot(bins_8,bin_freq_8,label='L=8')
plt.plot(bins_16,bin_freq_16,label='L=16')
plt.plot(bins_32,bin_freq_32,label='L=32')
plt.plot(bins_64,bin_freq_64,label='L=64')
plt.plot(bins_128,bin_freq_128,label='L=128')
plt.plot(bins_256,bin_freq_256,label='L=256')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('s')
plt.ylabel('$ P_N(s;L) $')
plt.grid()
plt.legend()
plt.show()
#%%
bins_4,bin_freq_4 = logbin(s_4_ss,scale=1.5,zeros=True)
bins_8,bin_freq_8 = logbin(s_8_ss,scale=1.5,zeros=True)
bins_16,bin_freq_16 = logbin(s_16_ss,scale=1.5,zeros=True)
bins_32,bin_freq_32 = logbin(s_32_ss,scale=1.5,zeros=True)
bins_64,bin_freq_64 = logbin(s_64_ss,scale=1.5,zeros=True)
bins_128,bin_freq_128 = logbin(s_128_ss,scale=1.5,zeros=True)
bins_256,bin_freq_256 = logbin(s_256_ss,scale=1.5,zeros=True)

plt.plot(bins_4,bin_freq_4,label='L=4')
plt.plot(bins_8,bin_freq_8,label='L=8')
plt.plot(bins_16,bin_freq_16,label='L=16')
plt.plot(bins_32,bin_freq_32,label='L=32')
plt.plot(bins_64,bin_freq_64,label='L=64')
plt.plot(bins_128,bin_freq_128,label='L=128')
plt.plot(bins_256,bin_freq_256,label='L=256')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
#%%
#Task 3a) ii)
#Data Collapse P(s;L)
#first find thegradient of the descent
first22_bins_256 = bins_256[:23]
first22_bin_freq_256 = bin_freq_256[:23]
log_bins_256 = np.log(first22_bins_256)
log_bin_freq_256 = np.log(first22_bin_freq_256)

def line(x,m,c):
    return m*x + c

popt,v = sci.curve_fit(line,log_bins_256,log_bin_freq_256)
m,c = popt
m_err = np.sqrt(v[0,0])

print('gradient = ',m,'+-',m_err , 'intercept = ',c)

tau_s = -m
mod_bin_freq_4 = bin_freq_4*(bins_4**tau_s)# mod means modified not modulus
mod_bin_freq_8 = bin_freq_8*(bins_8**tau_s)
mod_bin_freq_16 = bin_freq_16*(bins_16**tau_s)
mod_bin_freq_32 = bin_freq_32*(bins_32**tau_s)
mod_bin_freq_64 = bin_freq_64*(bins_64**tau_s)
mod_bin_freq_128 = bin_freq_128*(bins_128**tau_s)
mod_bin_freq_256 = bin_freq_256*(bins_256**tau_s)
plt.plot(bins_4,mod_bin_freq_4,label='L=4')
plt.plot(bins_8,mod_bin_freq_8,label='L=8')
plt.plot(bins_16,mod_bin_freq_16,label='L=16')
plt.plot(bins_32,mod_bin_freq_32,label='L=32')
plt.plot(bins_64,mod_bin_freq_64,label='L=64')
plt.plot(bins_128,mod_bin_freq_128,label='L=128')
plt.plot(bins_256,mod_bin_freq_256,label='L=256')#Collapsed in the y axis
plt.xlabel('s')
plt.ylabel('$ s^{t_s} P_N(s;L) $')
plt.legend()
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.show()
#%%
#collapsing in the x axis
D_vals = np.linspace(2,2.3,20)
for i in D_vals:
    mod_bins_4 = bins_4/4**i
    mod_bins_8 = bins_8/8**i
    mod_bins_16 = bins_16/16**i
    mod_bins_32 = bins_32/32**i
    mod_bins_64 = bins_64/64**i
    mod_bins_128 = bins_128/128**i
    mod_bins_256 = bins_256/256**i
    
    plt.plot(mod_bins_4,mod_bin_freq_4,label='L=4')
    plt.plot(mod_bins_8,mod_bin_freq_8,label='L=8')
    plt.plot(mod_bins_16,mod_bin_freq_16,label='L=16')
    plt.plot(mod_bins_32,mod_bin_freq_32,label='L=32')
    plt.plot(mod_bins_64,mod_bin_freq_64,label='L=64')
    plt.plot(mod_bins_128,mod_bin_freq_128,label='L=128')
    plt.plot(mod_bins_256,mod_bin_freq_256,label='L=256')
    plt.xlabel('s/Li')
    plt.ylabel('$ s^{t_s} P_N(s;L) $')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.xscale('log')
    plt.title(i)
    plt.show()
#%%
D = 2.0947
mod_bins_4 = bins_4/4**D
mod_bins_8 = bins_8/8**D
mod_bins_16 = bins_16/16**D
mod_bins_32 = bins_32/32**D
mod_bins_64 = bins_64/64**D
mod_bins_128 = bins_128/128**D
mod_bins_256 = bins_256/256**D

plt.plot(mod_bins_4,mod_bin_freq_4,label='L=4')
plt.plot(mod_bins_8,mod_bin_freq_8,label='L=8')
plt.plot(mod_bins_16,mod_bin_freq_16,label='L=16')
plt.plot(mod_bins_32,mod_bin_freq_32,label='L=32')
plt.plot(mod_bins_64,mod_bin_freq_64,label='L=64')
plt.plot(mod_bins_128,mod_bin_freq_128,label='L=128')
plt.plot(mod_bins_256,mod_bin_freq_256,label='L=256')
plt.xlabel('$s/L^{D}$')
plt.ylabel('$ s^{t_s} P_N(s;L) $')
plt.legend()
plt.grid()
plt.yscale('log')
plt.xscale('log')
#plt.title('Data Collapse for $P_N(s;L)$')
plt.show()
print('D =',D,', Tau_s =',tau_s)
#%%
#Task 3b
s_4_ss = np.array(s_4_ss)
s_8_ss = np.array(s_8_ss)
s_16_ss = np.array(s_16_ss)
s_32_ss = np.array(s_32_ss)
s_64_ss = np.array(s_64_ss)
s_128_ss = np.array(s_128_ss)
s_256_ss = np.array(s_256_ss)

s_4_ss = s_4_ss.astype('int64')
s_8_ss = s_8_ss.astype('int64')
s_16_ss = s_16_ss.astype('int64')
s_32_ss = s_32_ss.astype('int64')
s_64_ss = s_64_ss.astype('int64')
s_128_ss = s_128_ss.astype('int64')
s_256_ss = s_256_ss.astype('float64')
#%%
mom_1_4 = np.mean(s_4_ss)
mom_1_8 = np.mean(s_8_ss)
mom_1_16 = np.mean(s_16_ss)
mom_1_32 = np.mean(s_32_ss)
mom_1_64 = np.mean(s_64_ss)
mom_1_128 = np.mean(s_128_ss)
mom_1_256 = np.mean(s_256_ss)

mom_2_4 = np.mean(np.power(s_4_ss,2))
mom_2_8 = np.mean(np.power(s_8_ss,2))
mom_2_16 = np.mean(np.power(s_16_ss,2))
mom_2_32 = np.mean(np.power(s_32_ss,2))
mom_2_64 = np.mean(np.power(s_64_ss,2))
mom_2_128 = np.mean(np.power(s_128_ss,2))
mom_2_256 = np.mean(np.power(s_256_ss,2))

mom_3_4 = np.mean(np.power(s_4_ss,3))
mom_3_8 = np.mean(np.power(s_8_ss,3))
mom_3_16 = np.mean(np.power(s_16_ss,3))
mom_3_32 = np.mean(np.power(s_32_ss,3))
mom_3_64 = np.mean(np.power(s_64_ss,3))
mom_3_128 = np.mean(np.power(s_128_ss,3))
mom_3_256 = np.mean(np.power(s_256_ss,3))

mom_4_4 = np.mean(np.power(s_4_ss,4))
mom_4_8 = np.mean(np.power(s_8_ss,4))
mom_4_16 = np.mean(np.power(s_16_ss,4))
mom_4_32 = np.mean(np.power(s_32_ss,4))
mom_4_64 = np.mean(np.power(s_64_ss,4))
mom_4_128 = np.mean(np.power(s_128_ss,4))
mom_4_256 = np.mean(np.power(s_256_ss,4))
#%%
#Plotting moments as a function of L
mom_1 = [mom_1_4,mom_1_8,mom_1_16,mom_1_32,mom_1_64,mom_1_128,mom_1_256]
plt.plot(L_list,mom_1,'x',color='black')

mom_2 = [mom_2_4,mom_2_8,mom_2_16,mom_2_32,mom_2_64,mom_2_128,mom_2_256]
plt.plot(L_list,mom_2,'x',color='black')

mom_3 = [mom_3_4,mom_3_8,mom_3_16,mom_3_32,mom_3_64,mom_3_128,mom_3_256]
plt.plot(L_list,mom_3,'x',color='black')

mom_4 = [mom_4_4,mom_4_8,mom_4_16,mom_4_32,mom_4_64,mom_4_128,mom_4_256]
plt.plot(L_list,mom_4,'x',color='black')

def linear(log_L,m,log_c):
    log_mom = m*log_L + log_c
    return log_mom

popt_1,v_1 = sci.curve_fit(linear,np.log10(L_list),np.log10(mom_1))
m_1,log_c_1 = popt_1
err_m_1 = np.sqrt(v_1[0][0])
err_log_c_1 = np.sqrt(v_1[1][1])
x = np.linspace(4,256,500)
#plt.plot(x,linear(x,m_1,log_c_1), '--',color='red',label='k=1')

popt_2,v_2 = sci.curve_fit(linear,np.log10(L_list),np.log10(mom_2))
m_2,log_c_2 = popt_2
err_m_2 = np.sqrt(v_2[0][0])
err_log_c_2 = np.sqrt(v_2[1][1])
#plt.plot(x,linear(x,m_2,log_c_2), '--',color='blue',label='k=2')

popt_3,v_3 = sci.curve_fit(linear,np.log10(L_list),np.log10(mom_3))
m_3,log_c_3 = popt_3
err_m_3 = np.sqrt(v_3[0][0])
err_log_c_3 = np.sqrt(v_3[1][1])
#plt.plot(x,linear(x,m_3,log_c_3), '--',color='green',label='k=3')

popt_4,v_4 = sci.curve_fit(linear,np.log10(L_list),np.log10(mom_4))
m_4,c_4 = popt_4
err_m_4 = np.sqrt(v_4[0][0])
err_c_4 = np.sqrt(v_4[1][1])
#plt.plot(x,linear(x,m_4,c_4), '--',color='purple',label='k=4')

plt.plot(L_list,mom_1,'--',color='red', label='k=1')
plt.plot(L_list,mom_2,'--',color='blue', label='k=2')
plt.plot(L_list,mom_3,'--',color='green', label='k=3')
plt.plot(L_list,mom_4,'--',color='purple', label='k=4')

plt.yscale('log')
plt.xscale('log')
plt.xlabel('L')
plt.ylabel(r'$ \langle s^{k} \rangle $')
plt.grid()
plt.legend()
plt.show()
#%%
#plotting the gradient as a function of k
grads = [m_1,m_2,m_3,m_4]
ks = [1,2,3,4]
grad_errs = [err_m_1,err_m_2,err_m_3,err_m_4]
plt.plot(ks,grads,'x',color='black')

def line_fit(x,m,c):
    return m*x + c

popt,v = sci.curve_fit(line_fit,ks,grads)
m,c = popt
m_err_new = np.sqrt([v[0,0]])
c_err_new = np.sqrt([v[1,1]])

x = np.linspace(1,4,20)

plt.plot(x,line_fit(x,m,c),'--',color='black')

plt.xlabel('k')
plt.ylabel('m')
plt.grid()
plt.show()

print(m,m_err_new)
#%%
#finding tau_s
x=np.linspace(-4,4,100)
y=line_fit(x,m,c)
plt.plot(x,y)
plt.grid()
plt.xlim(0.4,0.6)

x_int = -c/m
x_int_rel_err = np.sqrt((c_err_new/c)**2 + (m_err_new/m)**2)
tau_s_2 = x_int+1
tau_s_2_err = tau_s_2*x_int_rel_err

print(tau_s_2,tau_s_2_err)