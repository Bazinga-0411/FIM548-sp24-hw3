Loopback Call
# SDE $d S_t=S_t\left(r d t+\sigma d B_t\right), \quad S_0=100$

with price discretely monitored m=10 loopback call options with strike K=110 and T=1

To price the option PT using the antithetic variate method with Monte Carlo simulation, follow these steps:

Generate N/2 independent standard normal random numbers Z1, Z2, ..., ZN/2.
For each Zi, create an antithetic pair (Zi, -Zi).
Calculate the corresponding asset prices at maturity T for each pair: S_T(Zi) = S0 * exp((r - 0.5*σ^2)T + σsqrt(T)Zi) S_T(-Zi) = S0 * exp((r - 0.5σ^2)T - σsqrt(T)*Zi)
Evaluate the payoff for each pair: P(Zi) = max(S_T(Zi) - K, 0) P(-Zi) = max(S_T(-Zi) - K, 0)
Calculate the average payoff for each antithetic pair: P_avg(i) = 0.5 * (P(Zi) + P(-Zi))
Estimate the option price by discounting the average of all P_avg(i): PT_estimate = exp(-r*T) * (1/N) * Σ P_avg(i)
Repeat the process for N ∈ {10000, 50000, 100000} sample paths.
Here's a table representing the findings:

To estimate the price of PT using stratified sampling with proportional allocation and optimal allocation, follow these steps:

(a) Proportional Allocation:

Divide the standard normal distribution N(0, 1) into the given intervals: (-∞, -0.8), [-0.8, -0.4), [-0.4, 0.4), [0.4, 0.8), [0.8, ∞).
Calculate the probability of each interval based on the standard normal distribution.
Allocate the number of sample paths proportionally to each interval based on the probabilities.
For each interval, generate the required number of sample paths using the inverse transform method to obtain the terminal value of the driving Brownian motion.
Use the Brownian bridge construction to simulate the remaining path of S for each sample path.
Calculate the payoff for each sample path and take the average.
Discount the average payoff to obtain the price estimate of PT.
Repeat the process for N ∈ {10000, 50000, 100000} sample paths and present the results in a table.
(b) Optimal Allocation:

Generate M = 1000 pilot sample paths using the standard Monte Carlo method.
Use the pilot sample paths to estimate the standard deviation of the payoff for each interval.
Calculate the optimal allocation of sample paths for each interval using the Neyman allocation formula: n_i = (N * σ_i) / (Σ σ_i) where n_i is the number of sample paths allocated to interval i, N is the total number of sample paths, and σ_i is the estimated standard deviation of the payoff for interval i.
Generate the required number of sample paths for each interval using the inverse transform method to obtain the terminal value of the driving Brownian motion.
Use the Brownian bridge construction to simulate the remaining path of S for each sample path.
Calculate the payoff for each sample path and take the average within each interval.
Calculate the stratified estimator by taking the weighted average of the interval averages, where the weights are the probabilities of each interval.
Discount the stratified estimator to obtain the price estimate of PT.
Repeat the process for N ∈ {10000, 50000, 100000} sample paths and present the results in a table.
Note: The specific numerical results will depend on the implementation and the random seed used. The tables should include the price estimates for each value of N and the corresponding standard errors.

By comparing the results from proportional allocation and optimal allocation, you can observe the potential improvement in the accuracy of the price estimates when using optimal allocation based on the estimated standard deviations from the pilot sample paths.
