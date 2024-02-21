#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Simulation of the dispersion swap

@author: Biwen Ling

This scipt can be used to simulate the payoff of the dispersion swap proposed in the paper titled
'Understanding the correlation risk premium'.
"""
import numpy as np
import matplotlib.pyplot as plt


# Multi-period binomial tree for two stocks.
u1 = np.log(1.4) 
d1 = np.log(0.6)
u2 = np.log(1.7) 
d2 = np.log(0.3)


# Consider T=10, rho_P(t) = 0.2. 
# Market situation 1: rho_Q1(t) = 0.05t - 0.95
# Market situation 2: rho_Q2(t) = -0.05t + 0.95
T = 10 
rho_Q1 = np.array([-0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45])
rho_Q2 = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45])
rho_P = np.empty(10)
rho_P.fill(0.2)
p_uu = 1/4*(1+rho_P)
p_dd = p_uu
p_ud = 1/2 - p_uu
p_du = p_ud

# Fixed leg: given by Expression (40).
Fixedleg_Q1 = np.sum(rho_Q1)*(u1-d1)*(u2-d2)/(4*T)+(u1+d1)*(u2+d2)/4
Fixedleg_Q2 = np.sum(rho_Q2)*(u1-d1)*(u2-d2)/(4*T)+(u1+d1)*(u2+d2)/4

# Simulate 10000 paths for the floating leg
N = 10000
sim = np.zeros((T, N))
elements = [u1*u2, u1*d2, d1*u2, d1*d2]
np.random.seed(3)
for i in range(T):
    probabilities = [p_uu[i], p_ud[i], p_du[i], p_dd[i]]
    sim[i] = np.random.choice(elements, N, p= probabilities)
RD_sim = np.sum(sim, axis = 0)

# Calculate the payoff for 10000 paths in two different market situations
Profit_Q1 = RD_sim - Fixedleg_Q1
Profit_Q2 = RD_sim - Fixedleg_Q2

#  Payoff histogram for longing the dispersion swap in Market situation 1
plt.figure()
plt.hist(Profit_Q1,bins=np.arange(-2, 4, 0.2))
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.ylim(0,1100)


#  Payoff histogram for longing the dispersion swap in Market situation 2
plt.figure()
plt.hist(Profit_Q2, bins=np.arange(-2, 4, 0.2))
plt.xlabel("Payoff")
plt.ylabel("Frequency")
plt.ylim(0,1100)