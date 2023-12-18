import numpy as np
import random as rd
import matplotlib.pyplot as plt

def SimulInversionBernouilli(P:float,N:int):
    U = np.random.random(N)
    X = np.zeros(N)
    
    for i in range(0,N):
        
        if( U[i]<P) : 
            
            X[i]=1
            
        else :
            
            X[i]=0


    plt.hist(X,             
             bins=2, 
             color='skyblue', 
             edgecolor='black',
             density=False,
             label="Simulation")
    
    plt.xlabel("V.A")
    plt.ylabel("Density")
    plt.title("Simulation by inversion")

    return None

def SimulInversionBinomial(P: float, N: int, n_simbernouilli: int):
    X_total = np.zeros(N) 
    
    for i in range(N):
        X = np.zeros(n_simbernouilli)
        U = np.random.random(n_simbernouilli)
        
        for j in range(n_simbernouilli):
            if(U[j] <P) : 
                X[j]=1
            else :
                X[j]=0


        X_total[i] = np.sum(X)

    plt.hist(X_total, color='skyblue', edgecolor='black', density=True, label="Simulation")
    plt.xlabel("Number of succes")
    plt.ylabel("Density")
    plt.title("Simulation of binomial distribution by Inversion method")
    plt.legend()

# Exemple d'utilisation

SimulInversionBinomial(0.25, 1000, 100)
plt.show()


SimulInversionBernouilli(1/2,100000)
    
plt.show()
