import numpy as np
import matplotlib.pyplot as plt


def BoxMullerCartesien(N:int):
    
    # creation of two sample of uniform randon value between 0 and 1
    U1 = np.random.uniform(0,1,N)
    U2 = np.random.uniform(0,1,N)
    
    # simulation and graph of gaussien pdf
    N1 = np.linspace(-3,3,N)
    sigma = 1
    mu = 0
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((N1 - mu) ** 2) / (2 * sigma ** 2))
    
    # compute X and Y (independant) by Box-Muller technique
    X = np.sqrt(-2.0*np.log(U1))*np.cos(2.0*np.pi*U2)
    Y = np.sqrt(-2.0*np.log(U1))*np.sin(2.0*np.pi*U2)
    
    
    # vizualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].hist(X, 
             density=True, 
             bins=25, 
             color='skyblue', 
             edgecolor='black', 
             alpha=0.7,
             label="Simulation")
    
    axes[0].plot(N1,
             pdf,
             label="N(0,1)")
    
    axes[1].hist(Y, 
             density=True, 
             bins=25, 
             color='red', 
             edgecolor='black', 
             alpha=0.7,
             label="Simulation")
    
    axes[1].plot(N1,
             pdf,
             label="N(0,1)")
    
    axes[0].set_title("Loi Normale centré réduite $X \sim \mathcal{N}(0,1)$ par Box-Muller")
    axes[0].legend()
    axes[0].set_xlim(-4,4)
    
    axes[1].set_title("Loi Normale centré réduite $Y \sim \mathcal{N}(0,1)$ par Box-Muller")
    axes[1].legend()
    axes[1].set_xlim(-4,4)
    
    plt.show()
    
    

def VisualisationBoxMuller(N):
    
    # tirage de v.a uniformément répartie
    U1 = np.random.uniform(0,1,N)
    U2 = np.random.uniform(0,1,N)

    
    #calcul des v.a indépendantes de N(0,1) par le procédé de BoxMuller
    Z1 = np.sqrt(-2.0*np.log(U1))*np.cos(2.0*np.pi*U2)
    Z2 = np.sqrt(-2.0*np.log(U1))*np.sin(2.0*np.pi*U2)
    
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(U1,U2,
                    color="red",
                    marker="o",
                    s=0.5)
    axes[0].set_title("Simulation de U1 et U2 -> U([0,1])")
    
    axes[1].scatter(Z1,Z2,
                    marker="o",
                    color="blue",
                    s=0.5,
                    label=" Z = (Z1,Z2) ")
    
    axes[1].set_title("Application de la transformation de BOX-MULLER sur U1 et U2")
    
    plt.grid()
    plt.legend()
    plt.show()
      
    
def BoxMullerPolaire(N):
    
    lambda1 = 0.5
    theta = np.random.uniform(0, 2 * np.pi,N)
    r = np.sqrt(np.random.uniform(0, 1,N))
    r1 = np.random.exponential(lambda1,N)

    # Coordonnées cartésiennes à partir des coordonnées polaires
    x_normale = r1 * np.cos(theta)
    y_normale = r1 * np.sin(theta)

    x_unif = r * np.cos(theta)
    y_unif = r * np.sin(theta)

    # Affichez le point (X, Y) sur le cercle unité
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    #Graphique loi normale par Box-Muller.
    axes[0].scatter(x_normale,
                    y_normale, 
                    color='blue',
                    marker='o',
                    s=0.5)
    
    axes[0].set_xlim(-4.0,4.0)
    axes[0].set_ylim(-4.0,4.0)
    axes[0].set_xlabel("Variable Normale X")
    axes[0].set_ylabel("Variable Normale Y")
    axes[0].grid(True)
    axes[0].set_title("Simulation d'un couple $(X,Y) \sim \mathcal{N}(0,1)$ par la méthode de Box-Muller")
    
    
    #Graphique loi Uniforme cercle unité (pour simulation de Box-Muller)
    axes[1].scatter(x_unif,
                    y_unif, 
                    color='blue', 
                    marker='o',
                    s=0.5)
    
    axes[1].set_title("Génération d'un couple $(X,Y) \sim \mathcal{U}(0,1)$")
    axes[1].set_xlabel("Variable Uniforme X")
    axes[1].set_ylabel("Variable Uniforme Y")
    axes[1].grid(True)
    plt.show()
    


    

affichage = 3
N = 10000
if (affichage==2):
    BoxMullerCartesien(N)
elif (affichage==3):
    VisualisationBoxMuller(N)
else:
    BoxMullerPolaire(N)




