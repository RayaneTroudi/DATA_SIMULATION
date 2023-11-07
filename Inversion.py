import numpy as np
import matplotlib.pyplot as plt
import random as rd 

def simulInversionLoiExp(lambda1:float, N:int):
    
    U = np.random.random(N)
    
    X = (-1.0/lambda1) * np.log(1.0-U)
    
    # Création de l'histogramme
    plt.hist(X, 
             density=True, 
             bins=30, 
             color='skyblue', 
             edgecolor='black', 
             alpha=0.7,
             label="Simulation")

    # Ajout d'étiquettes d'axe et d'un titre
    plt.xlabel('Quantiles X')
    plt.ylabel('Densité de probabilité')
    plt.title("Simulation par Inversion d'une loi Exponentielle" 
              " avec $\\lambda = \\frac{1}{4}$")
    
    
    tmp = np.linspace(0,40,1000)
    
    tmpf = lambda1*np.exp(-lambda1*tmp)
    
    plt.plot(tmp,
             tmpf,
             color='r',
             label="Théorique")
    
    plt.legend()
    
    plt.show();
    
    return None



def fx():
    
    X1 = np.linspace(0,2,1000)
    X2 = np.linspace(2,4,1000)
    
    Y1 = (1.0/4.0)*(X1)
    Y2 = (-1.0/4.0)*(X2 - 4.0)
    
    plt.plot(X1,Y1,
             label="$f(x) = \\frac{1}{4}x$",
             color="b")
    
    plt.plot(X2,Y2,
             label="f(x) = $\\frac{1}{4} \\dot (x-4)$",
             color="r")    
    
def Fx():
    
    X1 = np.linspace(0,2,500)
    X2 = np.linspace(2,4,500)
    
    Y1 = (1.0/8.0)*(X1**2)
    Y2 = 1.0/2.0 - (1.0/4.0)*(((X2-4)**2)/2.0-2)
    
    X = np.concatenate(X1,X2)
    Y = np.concatenate(Y1,Y2)
    
    plt.plot(X,Y)
    
    
def F_1_x(N:int):
    
    U = np.random.random(N)
    
    X = np.zeros(N)
    
    for i in range(0,N):
        
        if(U[i] >= 0 and U[i]<=1.0/2.0):
            
            X[i] = np.sqrt(8.0*U[i])
            
        else:
            X[i] = 2.0*(2-np.sqrt(2)*np.sqrt(1.0-U[i]))

    plt.hist(X,             
             bins=50, 
             color='skyblue', 
             edgecolor='black',
             density=True,
             label="Simulation")
    
    plt.xlabel("V.A")
    plt.ylabel("Densité")
    plt.title("Simulation par Inversion")

    fx() #densité de la fonction
    
    plt.legend()

    plt.show()
            
        
        
    
F_1_x(100000)

#simulInversionLoiExp(1/4,1000)