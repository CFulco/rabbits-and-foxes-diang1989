
# coding: utf-8

# # Rabbits and foxes
# 
# There are initially 400 rabbits and 200 foxes on a farm (but it could be two cell types in a 96 well plate or something, if you prefer bio-engineering analogies). Plot the concentration of foxes and rabbits as a function of time for a period of up to 600 days. The predator-prey relationships are given by the following set of coupled ordinary differential equations:
# 
# \begin{align}
# \frac{dR}{dt} &= k_1 R - k_2 R F \tag{1}\\
# \frac{dF}{dt} &= k_3 R F - k_4 F \tag{2}\\
# \end{align}
# 
# * Constant for growth of rabbits $k_1 = 0.015$ day<sup>-1</sup>
# * Constant for death of rabbits being eaten by foxes $k_2 = 0.00004$ day<sup>-1</sup> foxes<sup>-1</sup>
# * Constant for growth of foxes after eating rabbits $k_3 = 0.0004$ day<sup>-1</sup> rabbits<sup>-1</sup>
# * Constant for death of foxes $k_1 = 0.04$ day<sup>-1</sup>
# 
# Also plot the number of foxes versus the number of rabbits.
# 
# Then try also with 
# * $k_3 = 0.00004$ day<sup>-1</sup> rabbits<sup>-1</sup>
# * $t_{final} = 800$ days
# 
# *This problem is based on one from Chapter 1 of H. Scott Fogler's textbook "Essentials of Chemical Reaction Engineering".*
# 

# # Solving ODEs
# 
# *Much of the following content reused under Creative Commons Attribution license CC-BY 4.0, code under MIT license (c)2014 L.A. Barba, G.F. Forsyth. Partly based on David Ketcheson's pendulum lesson, also under CC-BY. https://github.com/numerical-mooc/numerical-mooc*
# 
# Let's step back for a moment. Suppose we have a first-order ODE $u'=f(u)$. You know that if we were to integrate this, there would be an arbitrary constant of integration. To find its value, we do need to know one point on the curve $(t, u)$. When the derivative in the ODE is with respect to time, we call that point the _initial value_ and write something like this:
# 
# $$u(t=0)=u_0$$
# 
# In the case of a second-order ODE, we already saw how to write it as a system of first-order ODEs, and we would need an initial value for each equation: two conditions are needed to determine our constants of integration. The same applies for higher-order ODEs: if it is of order $n$, we can write it as $n$ first-order equations, and we need $n$ known values. If we have that data, we call the problem an _initial value problem_.
# 
# Remember the definition of a derivative? The derivative represents the slope of the tangent at a point of the curve $u=u(t)$, and the definition of the derivative $u'$ for a function is:
# 
# $$u'(t) = \lim_{\Delta t\rightarrow 0} \frac{u(t+\Delta t)-u(t)}{\Delta t}$$
# 
# If the step $\Delta t$ is already very small, we can _approximate_ the derivative by dropping the limit. We can write:
# 
# $$\begin{equation}
# u(t+\Delta t) \approx u(t) + u'(t) \Delta t
# \end{equation}$$
# 
# With this equation, and because we know $u'(t)=f(u)$, if we have an initial value, we can step by $\Delta t$ and find the value of $u(t+\Delta t)$, then we can take this value, and find $u(t+2\Delta t)$, and so on: we say that we _step in time_, numerically finding the solution $u(t)$ for a range of values: $t_1, t_2, t_3 \cdots$, each separated by $\Delta t$. The numerical solution of the ODE is simply the table of values $t_i, u_i$ that results from this process.
# 

# # Euler's method
# *Also known as "Simple Euler" or sometimes "Simple Error".*
# 
# The approximate solution at time $t_n$ is $u_n$, and the numerical solution of the differential equation consists of computing a sequence of approximate solutions by the following formula, based on Equation (10):
# 
# $$u_{n+1} = u_n + \Delta t \,f(u_n).$$
# 
# This formula is called **Euler's method**.
# 
# For the equations of the rabbits and foxes, Euler's method gives the following algorithm that we need to implement in code:
# 
# \begin{align}
# R_{n+1} & = R_n + \Delta t \left(k_1 R_n - k_2 R_n F_n \right) \\
# F_{n+1} & = F_n + \Delta t \left( k_3 R_n F-n - k_4 F_n \right).
# \end{align}
# 

# In[1]:

#get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt


# In[2]:

import numpy as np
from matplotlib import pyplot as plt
# the kinetic constants given in the problem
k_1 = 0.015
k_2 = 0.00004
k_3 = 0.0004
k_4 = 0.04

t_start = 1  # first day
t_end = 800 # last day
n = 100000.  # the step size
t = np.linspace(t_start, t_end, n)
R = np.arange(n)  # the matrix for the number of rabbit
F = np.arange(n)  # the matrix for the number of fox
R[0] = 400                      # the initial value of rabbit
F[0] = 200                       # the initial value of fox

# Euler method 
for i in range(len(t)-1):
    delta_t = t[i+1] - t[i]
    R[i+1] = R[i] + delta_t * (k_1 * R[i] - k_2 * R[i] * F[i])
    F[i+1] = F[i] + delta_t * (k_3 * R[i] * F[i] - k_4 * F[i])

plt.xlabel('days', fontsize = 18)
plt.ylabel('animal numbers', fontsize = 18)
plt.plot(t, R, 'r', label='rabbits', linewidth = 1.5)
plt.plot(t, F, 'k', label='foxes', linewidth = 1.5)
legend = plt.legend(loc = 'upper right')
plt.show()


# In[2]:

#get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Kinetic parameters given in the problem
k_1 = 0.015
k_2 = 0.00004
k_3 = 0.0004
k_4 = 0.04

# initial condition
R_0 = 400.
F_0 = 200.

# days we are interested in
t_0 = 1.          # the first day
t_end = 600.      # the last day
n_E = 100000.      # steps in Euler method
n_KMC = 600.       # steps in KMC method

# Euler method
t = np.linspace(t_0, t_end, n_E)
R = np.arange(n_E)  # the matrix for the number of rabbit
F = np.arange(n_E)  # the matrix for the number of fox
R[0] = R_0                      # the initial value of rabbit
F[0] = F_0                       # the initial value of fox

# Euler method 
for i in range(len(t)-1):
    delta_t = t[i+1] - t[i]
    R[i+1] = R[i] + delta_t * (k_1 * R[i] - k_2 * R[i] * F[i])
    F[i+1] = F[i] + delta_t * (k_3 * R[i] * F[i] - k_4 * F[i])

plt.xlabel('Day', fontsize = 17)
plt.ylabel('Numbers of animal', fontsize = 17)
plt.title('Solution by Euler method', fontsize = 20)
plt.plot(t, R, 'r', label = 'Rabbits', linewidth  = 2)
plt.plot(t, F, 'k', label = 'Foxes', linewidth = 2)
plt.legend()
plt.show()
          
m = max(F)

print ('Solution via Euler method:')
print ("The max value of foxes:")
print (m)

for i in range (len(F)):
    if F[i] == m:
        t_max = (t_end-t_0)/(n_E-1) * i + 1 - t_0 
print ('The day when the number of foxes reach the maximum value:')
print (t_max)

def f (A, t):
    R_1, F_1 = A
    dRdt = k_1 * R_1 - k_2 * R_1 * F_1
    dFdt = k_3 * R_1 * F_1 - k_4 * F_1 
    return (dRdt, dFdt)

t = np.linspace (t_0, t_end, n_KMC)
A_0 = [R_0, F_0]
A_result = odeint(f, A_0, t)
R_1, F_1 = A_result.T 

plt.plot(t, R_1, label = 'Rabbits', linewidth  = 2)
plt.plot(t, F_1, label = 'Foxes', linewidth = 2)
plt.xlabel('Day', fontsize = 17)
plt.ylabel('Numbers of animal', fontsize = 17)
plt.title('Solution by odeint method', fontsize = 20)
plt.legend()
plt.show()
m = max(F_1)
# Looking for the maximum value of fox and on which day
print('Solution via odeint method:')
print ("The max value of foxes:")
print (m)
for i in range (len(F_1)):
    if F_1[i] == m:
        t_max = (t_end-t_0)/(n_KMC-1) * i + 1 - t_0 
print('The day when the number of foxes reach the maximum value:')
print(t_max)


# In[3]:

#get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

# Kintetic constants given in the problem
k_1 = 0.015
k_2 = 0.00004
k_3 = 0.0004
k_4 = 0.04

# Initial condition
R_ini = 400.0
F_ini = 200.0
t_ini = 0.0
f = 0.0 # the number of even that only foxes 
r = 0.0 # the number of even that both of rabbits and foxes died out

# while loop
j = 0
n = 20

# Create empty lists for the second fox peak and its corresponding time
f_peak = []
f_peak_t = []

while j < n:

# Create empty list for rabbits and foxes
    R = []
    F = []
    t = []
# Assign the initial condition in the empty list    
    R.append(R_ini)
    F.append(F_ini)
    t.append(t_ini)
# Start iteration here
    for i in range(50000):
        r_1 = k_1 * R [i]
        r_2 = k_2 * R [i] * F [i]
        r_3 = k_3 * R [i] * F [i]
        r_4 = k_4 * F [i]

        R_1 = r_1
        R_2 = r_1 + r_2
        R_3 = r_1 + r_2 + r_3
        Q = r_1 + r_2 + r_3 + r_4
        
        if Q == 0.0:
            j = j + 1
            r = r + 1    # the possibility that both rabbits and foxes die out
            break
        v = 1 - np.random.rand () #rids possibility of division by zero
        delta_t = ( 1 / Q ) * (np.log (1 / v))
        t.append ( t [i] + delta_t )
        u = np.random.rand()
        m = u * Q
        if m <= R_1:
            R.append (R [i] + 1)
            F.append (F [i] )
        else:
            if R_1 < m <= R_2:
                R.append(R[i] - 1)
                F.append(F[i])
            else:
                if R_2 < m <= R_3:
                    R.append(R[i])
                    F.append(F[i] + 1)
                else: #No need to have last if statement. If it gets to this point, it will be true by default
                    R.append(R[i])
                    F.append(F[i] - 1)
        if t[-1] >=600:
            j = j + 1
            break
    if F [i] == 0:
        f  += 1
    # plot figure that foxes die out
        plt.figure (1)
        plt.plot(t, R, 'r')
        plt.plot(t, F, 'k')
        plt.title ('Foxes die out before 600 days')
        plt.xlabel ('days')
        plt.ylabel ('population')
        plt.legend(['Rabbits', 'Foxes'])
    else:
        plt.figure (2)
    # plot the figure that foxes survive
        fig_2 = plt.figure (2)
        plt.plot(t, R, 'r')
        plt.plot(t, F, 'k')
        plt.title ('Foxes survive before 600 days')
        plt.xlabel ('days')
        plt.ylabel ('population')
        plt.legend(['Rabbits', 'Foxes'])
        
# Looking for the second peaks 
        F_sp = np.zeros_like (np.arange (len(F)))
        T = np.zeros_like (np.arange (len(F)))
        a = int (round (0.5 * len (F)))
        for m in range (a, len(F)):
            F_sp [m] = F [m]
            T [m] = t [m]     
        f_peak.append (np.amax (F_sp))
        f_peak_t.append (T [np.argmax (F_sp)])   
plt.show()

# Possibility
p = round(f / j * 100, 2)  #the possibility that only foxes die out
p_1 = round(r / j * 100, 2) # the possibility that both die out


# Expected second peak
ave_peak = int (round (np.average(f_peak)))
ave_t = int ( round (np.average(f_peak_t)))

# Interquartile range 
IQR1_peak = np.percentile(f_peak, 25)
IQR3_peak = np.percentile(f_peak, 75)
IQR1_t = np.percentile(f_peak_t, 25)
IQR3_t = np.percentile(f_peak_t, 75)


print ("The results are based on {} iterations:". format (n))
print ("The probability that the foxes die out before 600 days are complete: {} %.". format (p))
print ("The probability that both die out before 600 days are complete: {} %.". format (p_1))
print ("The expected location of the second peak in foxes: {} foxes at {} day.". format (ave_peak, ave_t,))
print ("The interquartile range of the second peak in foxes: {} - {}, foxes at {} - {} days.". format(IQR1_peak, IQR3_peak,IQR1_t,IQR3_t,))


# In[ ]:

# What I learned from this assignment
# 1. It is interesting to learn KMC method, which is based on the possibility of kinetic events.
#    Based on the plot, it seems like there is a little connection between KMC and Euler & odeint method.
#    (just guess...)
# 2. In the KMC "for" loop, Q is occasionally equal to "0.0", which means the both rabbits and foxes 
#    die out at the same time. However, in the previous coding, I was trying to apply a "while" loop 
#    to void Q equal to "0.0", which takes much more time and it is not accurate
# 3. It might be good to learn how to improve runing time.


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



