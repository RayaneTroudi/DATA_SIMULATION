import numpy as np
import matplotlib.pyplot as plt

# Modifier la taille par défaut de toutes les figures
plt.rcParams['figure.figsize'] = [10, 8]

def SimulInversionBernouilli(P: float, N: int):
    U = np.random.random(N)
    X = np.zeros(N)
    
    for i in range(N):        
        if U[i] < P:            
            X[i] = 1
        else:            
            X[i] = 0

    plt.subplot(2, 1, 1)  # 2 lignes, 1 colonne, position 1
    bins = [-0.5, 0.5, 1.5]
    plt.hist(X, bins=bins, color='skyblue', edgecolor='black', density=True)
    plt.xlabel("V.A")
    plt.ylabel("Density")
    plt.title("Simulation by Inversion of Bernoulli Sample")
    plt.xticks([0, 1])

def SimulInversionBinomial(P: float, N: int, n_simbernouilli: int):
    X_total = np.zeros(N) 
    
    for i in range(N):
        X = np.zeros(n_simbernouilli)
        U = np.random.random(n_simbernouilli)
        
        for j in range(n_simbernouilli):
            if U[j] < P: 
                X[j] = 1
            else :
                X[j] = 0

        X_total[i] = np.sum(X)

    plt.subplot(2, 1, 2)  # 2 lignes, 1 colonne, position 2
    plt.hist(X_total, color='skyblue', edgecolor='black', density=True)
    plt.xlabel("Number of Successes")
    plt.ylabel("Density")
    plt.title("Simulation of Binomial Distribution by Inversion Method")

# Exemple d'utilisation
SimulInversionBinomial(0.25, 1000, 100)
SimulInversionBernouilli(1/2, 100000)

plt.tight_layout() 
plt.show()
