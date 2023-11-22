import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, ksone



def Fn_x(N:int,X:list, x_i:float):
    """Compute the F_n(x) for value x_i

    Args:
        N (int): size of X : must be the number of elements in X
        X (list): np.array of randoms values that we want to find distribution
        x_i (float) : x_i that we want to compute F_n(x_i)
        
    """
    # return the index where the value x_i must be insert to keep the right order
    index = np.searchsorted(X, x_i)
    
    Fn_x = 0
 
    if (index == 0):
        Fn_x = 0.0
    
    elif (index == N-1):
        Fn_x = 1.0
    else:
        Fn_x = float((float(index))/float(N))
        
    return Fn_x


def VisualKolmogorovSmirnov(X):
    
    size_X = len(X)
    X_sort = np.sort(X)
    
    K = np.zeros(size_X)
    
    mu = 0
    sigma = 1
    
    # Compute the statistic K for each j
    for j in range(1, size_X):
        
        # compute the cdf for each j
        cdf_theorique = norm.cdf(X_sort[j], mu, sigma)
        
        # Compute the statistic K
        K[j] = np.maximum(j/size_X - cdf_theorique, cdf_theorique - (j - 1) / size_X)
    
    # add extremities
    K[0] = norm.cdf(X_sort[0], mu, sigma)  # Aucun point avant le premier, donc c'est simplement la CDF théorique
    K[-1] = 1 - norm.cdf(X_sort[-1], mu, sigma)  # Pour le dernier point, c'est 1 moins la CDF théorique
    
    return np.max(K) # return max

# ________________ BEGINNING OF KS-TEST ________________ 
 
n = 1000
mu = 0
sigma = 1

# random value of gaussien distribution
X = np.random.randn(n)
# drow the theorical cumulative function
x = np.linspace(-4,4,n)
Fx = norm.cdf(x,mu,sigma)

#compute the approximate cdf
approx_Fx_K = np.zeros(n)
for i in range(0,n):
    approx_Fx_K[i] = Fn_x(n,np.sort(X),x[i])
    
#set the risk alpha 
alpha = 0.05
#compute thorical critical value in KS-table
K_th = ksone.ppf(1 - alpha/2, n)
#compute empirical value
K = VisualKolmogorovSmirnov(X)
#make decision
if(K < K_th):
    print("Les données suivent une loi Normale u seuil alpha = 5% ( K = ",round(K,5),") et ( Dn = ",round(K_th,5)," )\n")
else:
    print("Les données ne suivent pas une loi Normale")

# vizualisation
plt.plot(x,approx_Fx_K,
         label="Cdf approximée par KS")
plt.plot(x,Fx,
         label="Cdf théorique d'une $\mathcal{N}(0,1)$")
plt.title("Méthode de Kolmogorov-Smirnov pour une  $\mathcal{N}(0,1)$")
plt.legend()
plt.grid()
plt.show()
    