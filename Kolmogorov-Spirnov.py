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


def GetKolmogorovSmirnovStatistic(X):
    
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
 
n = 10
n1 = 50
n2 = 100
n3 = 1000


mu = 0
sigma = 1
x1 = np.linspace(-4,4,1000)
Fx = norm.cdf(x1,mu,sigma)


# random value of gaussien distribution
X = np.random.randn(n)
X1 = np.random.randn(n1)
X2 = np.random.randn(n2)
X3 = np.random.randn(n3)




# drow the theorical cumulative function
x = np.linspace(-4,4,n)
xx = np.linspace(-4,4,n1)
xxx =np.linspace(-4,4,n2)
xxxx =np.linspace(-4,4,n3)



#compute the approximate cdf
approx_Fx_K = np.zeros(n)
approx_Fx_K1 = np.zeros(n1)
approx_Fx_K2 = np.zeros(n2)
approx_Fx_K3 = np.zeros(n3)

for i in range(0,n):
    approx_Fx_K[i] = Fn_x(n,np.sort(X),x[i])

for i in range(0,n1):
    approx_Fx_K1[i] = Fn_x(n1,np.sort(X1),xx[i])
    
for i in range(0,n2):
    approx_Fx_K2[i] = Fn_x(n2,np.sort(X2),xxx[i])
    
for i in range(0,n3):
    approx_Fx_K3[i] = Fn_x(n3,np.sort(X3),xxxx[i])
    
    
    
    
#set the risk alpha 
alpha = 0.05
#compute thorical critical value in KS-table
K_th = ksone.ppf(1 - alpha/2, n)
#compute empirical value
K = GetKolmogorovSmirnovStatistic(X)
#make decision
if(K < K_th):
    print("Les données suivent une loi Normale u seuil alpha = 5% ( K = ",round(K,5),") et ( Dn = ",round(K_th,5)," )\n")
else:
    print("Les données ne suivent pas une loi Normale")

# vizualisation


fig, axes = plt.subplots(nrows=2, ncols=2)

# Plot pour N = 10
axes[0, 0].step(x, approx_Fx_K)
axes[0, 0].plot(x1, Fx)
axes[0, 0].set_title("N = 10")
axes[0, 0].grid(True)

# Plot pour N = 50
axes[0, 1].step(xx, approx_Fx_K1)
axes[0, 1].plot(x1, Fx)
axes[0, 1].set_title("N = 50")
axes[0, 1].grid(True)

# Plot pour N = 100
axes[1, 0].step(xxx, approx_Fx_K2)
axes[1, 0].plot(x1, Fx)
axes[1, 0].set_title("N = 100")
axes[1, 0].grid(True)

# Plot pour N = 1000
axes[1, 1].step(xxxx, approx_Fx_K3)
axes[1, 1].plot(x1, Fx)
axes[1, 1].set_title("N = 1000")
axes[1, 1].grid(True)

# Ajoutez un titre général à la figure
fig.suptitle("Kolmogorov-Smirnov Test on a sample X ~ N(0,1)")

# Ajustez l'espacement entre les sous-graphiques
plt.tight_layout()

# Affichez la figure
plt.show()
    