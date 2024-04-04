#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
T = 1.0
S0=100
K=110
r=0.05
sigma = 0.2
m=100
N=1000

def generate_paths(N, m):
    dt = T / m
    paths = np.zeros((N, m+1))
    paths[:, 0] = S0
    for i in range(1, m+1):
        z = np.random.standard_normal(N)
        paths[:, i] = paths[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

#%%
def problem_1_a():
    paths = generate_paths(N,m)
    ST=paths[:,-1]
    max_ST_K = np.maximum(ST-K,0)
    paths = generate_paths(N, m)
    ST = paths[:, -1]
    max_ST_K = np.maximum(ST - K, 0)

    GT = np.mean(paths, axis=1)**(1/m)  
    max_GT_K = np.maximum(GT - K, 0)
    # Scatter plots
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.scatter(ST, max_ST_K)
    plt.xlabel('ST')
    plt.ylabel('max(ST-K, 0)')
    plt.title(f'Correlation: {np.corrcoef(ST, max_ST_K)[0,1]:.3f}')

    plt.subplot(1,3,2) 
    plt.scatter(ST, GT)
    plt.xlabel('ST')
    plt.ylabel('GT')  
    plt.title(f'Correlation: {np.corrcoef(ST, GT)[0,1]:.3f}')

    plt.subplot(1,3,3)
    plt.scatter(GT, max_GT_K)
    plt.xlabel('GT')
    plt.ylabel('max(GT-K, 0)')
    plt.title(f'Correlation: {np.corrcoef(GT, max_GT_K)[0,1]:.3f}')

    plt.tight_layout()


def monte_carlo_vanilla(N, m):
    paths = generate_paths(N, m)
    payoff = np.maximum(paths[:, -1] - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

def monte_carlo_cv(N, m):
    paths = generate_paths(N, m)
    ST = paths[:, -1]
    GT = np.mean(paths, axis=1)**(1/m)
    
    cov = np.cov(ST, GT)[0,1]
    var = np.var(GT)
    b = cov / var
    
    payoff = np.maximum(ST - K, 0)
    cv = b * (GT - np.exp(r*T) * S0)
    
    return np.exp(-r * T) * np.mean(payoff - cv)

def problem_1_b():
    for N in [10000, 50000, 100000]:
        price_vanilla = monte_carlo_vanilla(N, m)
        std_vanilla = np.sqrt(np.var(np.maximum(generate_paths(N, m)[:,-1] - K, 0)) / N) * np.exp(-r * T)
        
        price_cv = monte_carlo_cv(N, m)
        errors = np.maximum(generate_paths(N, m)[:,-1] - K, 0) - price_cv 
        std_cv = np.sqrt(np.var(errors) / N) * np.exp(-r * T)
        
        print(f"N = {N}")
        print(f"Vanilla MC Price: {price_vanilla:.3f}, Std: {std_vanilla:.3f}")  
        print(f"CV MC Price: {price_cv:.3f}, Std: {std_cv:.3f}")
        print()
# %%
problem_1_a()
problem_1_b()
# %%
def generate_antithetic_paths(N, m):
    dt = T / m
    paths_plus = np.zeros((N//2, m+1))
    paths_minus = np.zeros((N//2, m+1))
    paths_plus[:, 0] = S0
    paths_minus[:, 0] = S0
    
    for i in range(1, m+1):
        z = np.random.standard_normal(N//2)
        paths_plus[:, i] = paths_plus[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        paths_minus[:, i] = paths_minus[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt - sigma * np.sqrt(dt) * z)
        
    return np.concatenate((paths_plus, paths_minus), axis=0)

def monte_carlo_antithetic(N, m):
    paths = generate_antithetic_paths(N, m)
    payoff = np.maximum(paths[:, -1] - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

results = []

for N in [10000, 50000, 100000]:
    price_vanilla = monte_carlo_vanilla(N, m)
    std_vanilla = np.sqrt(np.var(np.maximum(generate_paths(N, m)[:,-1] - K, 0)) / N) * np.exp(-r * T)
    
    price_antithetic = monte_carlo_antithetic(N, m)
    errors = np.maximum(generate_antithetic_paths(N, m)[:,-1] - K, 0) - price_antithetic
    std_antithetic = np.sqrt(np.var(errors) / N) * np.exp(-r * T)
    
    results.append([N, price_vanilla, std_vanilla, price_antithetic, std_antithetic])

results = pd.DataFrame(results, columns=['N', 'Vanilla Price', 'Vanilla Std', 'Antithetic Price', 'Antithetic Std'])
print(results.to_string(index=False))

# %%
