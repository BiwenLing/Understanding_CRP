"""
Example: The multi-period model (presented in Section 3.3) 

@author: Biwen Ling

This script can be used to reproduce the numerical results in Section 3.3 of the paper titled 
'Understanding the Correlation Risk Premium'.

Additionally, the code can be used to calculate implied/realized correlations and implied/realized volatilities
when the underlying stock prices follow binomial tree model.
"""
import math
import itertools as it
import numpy as np
import matplotlib.pyplot as plt


# European call option prices with strike K = 300 and different maturities t.
call_option_price = [71.72, 98.03, 109.89, 127.91, 144.79, 154.04, 162.56, 174.16, 182.13, 187.13]
K = 300


# beta(x) is used to model the real-world correlation at time x.
def beta(x):
    return 0.92 - 0.08*x


# The payoff of the European call option with maturity t and strike K is max((S(t)-K),0)
def payoff(x, strike):
    if x-strike > 0:
        return x-strike
    else:
        return 0


# Future times t = 1,2, ..., T.
t = range(1, 11)
T = 10


# Multi-period binomial tree for two stocks.
u1 = np.log(1.4) 
d1 = np.log(0.6)
u2 = np.log(1.7) 
d2 = np.log(0.3)


# The prices of 2 different assets and the price of the market index.
S_1 = np.zeros((T+1, T+1))
S_2 = np.zeros((T+1, T+1))
# The time-0 price of the first stock S_1 and the second stock S_2. 
S_1[0, 0] = 100
S_2[0, 0] = 200
# The time-t price of the first stock S_1 and the second stock S_2. 
for i in t:
    for j in range(i+1):
        S_1[i, j] = S_1[0, 0]*np.exp(u1)**(i-j)*np.exp(d1)**j
        S_2[i, j] = S_2[0, 0]*np.exp(u2)**(i-j)*np.exp(d2)**j


# Real-world correlations rho_P(t)
rho_P = np.zeros(10)
for i in t:
    rho_P[i-1] = beta(i)
  
    
# Joint probabilities under real-world measure P
p_uu = np.zeros(T)
p_dd = np.zeros(T)
p_ud = np.zeros(T)
p_du = np.zeros(T)
for i in t:
    p_uu[i-1] = 0.25*(1+beta(i))
    p_ud[i-1] = 0.25*(1-beta(i))

p_du = p_ud
p_dd = p_uu



# Possible paths for stock prices
def stock_price_trace(time, down_count):
    stock_trace = np.array(list(it.product(range(2), repeat=time)))
    selected_index = np.where(np.sum(stock_trace, axis=1) == time-down_count)
    stock_trace = stock_trace[selected_index]
    return stock_trace


# Compute the real-world volatilities sigma_P(t) = sqrt(Var_P(log(S(t)/S(t-1))))
sigma_P = np.zeros(T)

# First find the time-1 price for the stock index S, which is given by S(1)=S_1(1)+S_2(1)
S_at_time1 = np.array([S_1[0, 0]*np.exp(u1) + S_2[0, 0]*np.exp(u2), S_1[0, 0]*np.exp(u1) + S_2[0, 0]*np.exp(d2), S_1[0, 0]*np.exp(d1) + S_2[0, 0]*np.exp(u2), S_1[0, 0]*np.exp(d1) + S_2[0, 0]*np.exp(d2)])
# Calculate the forward return of the stock index at time-1, which is given by R(1)=log(S(1)/S(0))
forward_return_time1 = np.log(S_at_time1)-np.log(S_1[0, 0] + S_2[0, 0])
# time-1 real-world joint probabilities
p_time1 = np.array([p_uu[0], p_ud[0], p_du[0], p_dd[0]])
# time-1 real-world expectation of the forward return
E_P_time1 = np.dot(forward_return_time1, p_time1)
# time-1 sigma_p, sigma_p(1)
sigma_P[0] = math.sqrt(np.dot(np.square(forward_return_time1 - E_P_time1), p_time1))


# time-2 to time-10 sigma_p
for i in range(2, T+1):
    conditional_S = np.zeros((i, i))
    p_conditional_S = np.zeros((i, i))
    conditional_expected_forward_return = np.zeros((i, i))
    conditional_forward_return_variance = np.zeros((i, i))
    for j in range(i):
        for k in range(i):
            conditional_S1 = S_1[i-1, j]
            conditional_S2 = S_2[i-1, k]
            conditional_S[j, k] = conditional_S1 + conditional_S2
            S_next = np.array([conditional_S1 * np.exp(u1) + conditional_S2 * np.exp(u2), conditional_S1 * np.exp(u1) + conditional_S2 * np.exp(d2), conditional_S1 * np.exp(d1) + conditional_S2 * np.exp(u2), conditional_S1 * np.exp(d1) + conditional_S2 * np.exp(d2)])
            conditional_forward_return = np.log(S_next)-np.log(conditional_S[j, k])
            p_next = np.array([p_uu[i-1], p_ud[i-1], p_du[i-1], p_dd[i-1]])
            conditional_expected_forward_return[j, k] = np.dot(conditional_forward_return, p_next)
            conditional_forward_return_variance[j, k] = np.dot(np.square(conditional_forward_return - conditional_expected_forward_return[j, k]), p_next)
            conditional_S1_trace = stock_price_trace(i-1, j)
            conditional_S2_trace = stock_price_trace(i-1, k)
            for s1_trace in conditional_S1_trace:
                for s2_trace in conditional_S2_trace:
                    same_move_idx = np.where(s1_trace - s2_trace == 0)
                    diff_move_idx = np.where(s1_trace - s2_trace != 0)
                    same_move_prob = np.product(p_uu[same_move_idx])
                    diff_move_prob = np.product(p_ud[diff_move_idx])
                    p_conditional_S[j, k] = p_conditional_S[j, k] + same_move_prob * diff_move_prob
    E_var = np.sum(p_conditional_S * conditional_forward_return_variance)
    E_E = np.sum(p_conditional_S * conditional_expected_forward_return)
    Var_E = np.sum((conditional_expected_forward_return - E_E) * (conditional_expected_forward_return - E_E) * p_conditional_S)
    sigma_P[i-1] = math.sqrt(E_var + Var_E)


# Using call option prices to get implied correlations rho_Q(t).
rho_Q = np.zeros(T)
q_uu = np.zeros(T)
q_ud = np.zeros(T)
q_du = np.zeros(T)
q_dd = np.zeros(T)

# time-1 payoff for the index call option C(1,K)
P1_time1 = payoff(S_at_time1[0], K)
P2_time1 = payoff(S_at_time1[1], K)
P3_time1 = payoff(S_at_time1[2], K)
P4_time1 = payoff(S_at_time1[3], K)
# time-1 implied correlation rho_Q(1)
rho_Q[0] = (4*call_option_price[0]-(P1_time1+P2_time1+P3_time1+P4_time1))/(P1_time1-P2_time1-P3_time1+P4_time1)
# time-1 risk-neutral joint probabilities
q_uu[0] = 1/4*(1+rho_Q[0])
q_dd[0] = q_uu[0]
q_ud[0] = 1/4*(1-rho_Q[0])
q_du[0] = q_ud[0]
# time-2 to time-10 implied correlations
for i in range(2, T+1):
    q_conditional_S = np.zeros((i, i))
    P1 = np.zeros((i, i))
    P2 = np.zeros((i, i))
    P3 = np.zeros((i, i))
    P4 = np.zeros((i, i))
    for j in range(i):
        for k in range(i):
            conditional_S1 = S_1[i-1, j]
            conditional_S2 = S_2[i-1, k]
            S_next = np.array([conditional_S1 * np.exp(u1) + conditional_S2 * np.exp(u2), conditional_S1 * np.exp(u1) + conditional_S2 * np.exp(d2), conditional_S1 * np.exp(d1) + conditional_S2 * np.exp(u2), conditional_S1 * np.exp(d1) + conditional_S2 * np.exp(d2)])
            P1[j, k] = payoff(S_next[0], K)
            P2[j, k] = payoff(S_next[1], K)
            P3[j, k] = payoff(S_next[2], K)
            P4[j, k] = payoff(S_next[3], K)
            conditional_S1_trace = stock_price_trace(i-1, j)
            conditional_S2_trace = stock_price_trace(i-1, k)
            for s1_trace in conditional_S1_trace:
                for s2_trace in conditional_S2_trace:
                    same_move_idx = np.where(s1_trace - s2_trace == 0)
                    diff_move_idx = np.where(s1_trace - s2_trace != 0)
                    same_move_prob = np.product(q_uu[same_move_idx])
                    diff_move_prob = np.product(q_ud[diff_move_idx])
                    q_conditional_S[j, k] = q_conditional_S[j, k] + same_move_prob * diff_move_prob
    factor1 = P1+P2+P3+P4
    factor2 = P1-P2-P3+P4
    rho_Q[i-1] = (4*call_option_price[i-1]-np.sum(q_conditional_S*factor1))/np.sum(q_conditional_S*factor2)
    q_uu[i-1] = 1/4*(1+rho_Q[i-1])
    q_dd[i-1] = q_uu[i-1]
    q_ud[i-1] = 1/4*(1-rho_Q[i-1])
    q_du[i-1] = q_ud[i-1]


    
# Same method to compute sigma_Q(t)
# time-1 sigma_Q
sigma_Q = np.zeros(T)
q_time1 = np.array([q_uu[0], q_ud[0], q_du[0], q_dd[0]])
E_Q_time1 = np.dot(forward_return_time1, q_time1)
sigma_Q[0] = math.sqrt(np.dot(np.square(forward_return_time1 - E_Q_time1), q_time1))
# time-2 to time-10 sigma_Q
for i in range(2, T+1):
    conditional_S = np.zeros((i, i))
    q_conditional_S = np.zeros((i, i))
    q_conditional_expected_forward_return = np.zeros((i, i))
    q_conditional_forward_return_variance = np.zeros((i, i))
    for j in range(i):
        for k in range(i):
            conditional_S1 = S_1[i-1, j]
            conditional_S2 = S_2[i-1, k]
            conditional_S[j, k] = conditional_S1 + conditional_S2
            S_next = np.array([conditional_S1 * np.exp(u1) + conditional_S2 * np.exp(u2), conditional_S1 * np.exp(u1) + conditional_S2 * np.exp(d2), conditional_S1 * np.exp(d1) + conditional_S2 * np.exp(u2), conditional_S1 * np.exp(d1) + conditional_S2 * np.exp(d2)])
            conditional_forward_return = np.log(S_next)-np.log(conditional_S[j, k])
            q_next = np.array([q_uu[i-1], q_ud[i-1], q_du[i-1], q_dd[i-1]])
            q_conditional_expected_forward_return[j, k] = np.dot(conditional_forward_return, q_next)
            q_conditional_forward_return_variance[j, k] = np.dot(np.square(conditional_forward_return - q_conditional_expected_forward_return[j, k]), q_next)
            conditional_S1_trace = stock_price_trace(i-1, j)
            conditional_S2_trace = stock_price_trace(i-1, k)
            for s1_trace in conditional_S1_trace:
                for s2_trace in conditional_S2_trace:
                    same_move_idx = np.where(s1_trace - s2_trace == 0)
                    diff_move_idx = np.where(s1_trace - s2_trace != 0)
                    same_move_prob = np.product(p_uu[same_move_idx])
                    diff_move_prob = np.product(p_ud[diff_move_idx])
                    q_conditional_S[j, k] = q_conditional_S[j, k] + same_move_prob * diff_move_prob
    E_Q_var = np.sum(q_conditional_S * q_conditional_forward_return_variance)
    E_Q_E = np.sum(q_conditional_S * q_conditional_expected_forward_return)
    Var_Q_E = np.sum((q_conditional_expected_forward_return - E_Q_E) * (q_conditional_expected_forward_return - E_Q_E) * q_conditional_S)
    sigma_Q[i-1] = math.sqrt(E_Q_var + Var_Q_E)





# plots:  rho_P VS rho_Q
plt.plot(t, rho_P, marker='o', ls='solid')
plt.plot(t, rho_Q, marker='s', markersize=3, ls='dashed')
plt.xlabel('time')
plt.ylabel('correlation')
plt.ylim(0, 1)
plt.legend(['realized correlation', 'implied correlation'], fontsize="9", loc='upper right')


# plots:  the correlation gap rho_P - rho_Q
print(rho_Q)
plt.plot(t, rho_P - rho_Q, marker='o', ls='solid')
plt.xlabel('time')
plt.ylabel('correlation gap')
label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
plt.xticks(t, label)































