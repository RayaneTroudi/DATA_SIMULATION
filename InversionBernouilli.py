import numpy as np
import random as rd
import matplotlib.pyplot as plt

def SimulInversionBernouilli(P:float,N:int):
    U = np.random.random(N)
    X = np.zeros(N)
    
    for i in range(0,N):
        
        if(U[i] >= 0 and U[i]<1-P) : 
            
            X[i]=0
            
        else :
            
            X[i]=1


    plt.hist(X,             
             bins=2, 
             color='skyblue', 
             edgecolor='black',
             density=False,
             label="Simulation")
    
    plt.xlabel("V.A")
    plt.ylabel("Densité")
    plt.title("Simulation par Inversion")

    return None

def SimulInversionBinomial(P: float, N: int, n_simbernouilli: int):
    X_total = np.zeros(N) 
    
    for i in range(N):
        X = np.zeros(n_simbernouilli)
        U = np.random.random(n_simbernouilli)
        
        for j in range(n_simbernouilli):
            if U[j] <= P:
                X[j] = 1
            else:
                X[j] = 0

        X_total[i] = np.sum(X)

    plt.hist(X_total, color='skyblue', edgecolor='black', density=True, label="Simulation")
    plt.xlabel("Nombre de succès")
    plt.ylabel("Densité")
    plt.title("Simulation d'une distribution binomiale par Inversion")
    plt.legend()

# Exemple d'utilisation

SimulInversionBinomial(0.5, 1000, 11)
plt.show()


SimulInversionBernouilli(1/2,100000)
    
plt.show()
