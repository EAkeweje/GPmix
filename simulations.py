__all__ = ['generate_data']

import numpy as np
import matplotlib.pyplot as plt
import scipy
import skfda

def generate_data(scenario : str):
    '''
    Generate synthetic datasets from any of the simulation scenarios A to L.

    Parameters
    ----------
    scenario : str
        Specifies the simulation scenario from which to generate the synthetic dataset.
        Choose from scenarios A to L.

    Returns
    -------
    dataset : (FDataGrid object, array-like)
        The synthetic dataset, the functional data (as FDataGrid object) and label (as array-like), generated based on the specified simulation scenario.
    '''

    data_dict = {
        'a' : get_a(),
        'b' : get_b(),
        'c' : get_c(),
        'd' : get_d(),
        'e' : get_e(),
        'f' : get_f(),
        'g' : get_g(),
        'h' : get_h(),
        'i' : get_i(),
        'j' : get_j(),
        'k' : get_k(),
        'l' : get_l()
    }
    return data_dict[scenario.lower()]


def matern_kernel(x, y, lengthscale=1.0, nu=1.5):
    x = np.asarray(x)
    y = np.asarray(y)
    dists = np.abs(x[:, None] - y[None, :])

    if nu == 0.5:
        K = np.exp(-dists / lengthscale)
    elif nu == 1.5:
        K = dists * np.sqrt(3) / lengthscale
        K = (1. + K) * np.exp(-K)
    elif nu == 2.5:
        K = dists * np.sqrt(5) / lengthscale
        K = (1. + K + K**2 / 3.0) * np.exp(-K)
    else:  # general case
        K = dists * np.sqrt(2 * nu) / lengthscale
        K = scipy.special.kv(nu, K) * 2**(1. - nu) / scipy.special.gamma(nu)
        K[dists == 0.0] = 1.0
    return K.squeeze()

def exponentiated_quadratic(xa, xb):
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

################################### Simulation scenario A ###################################
def get_a():
    n_pts = 101
    t = np.linspace(0,1,n_pts)

    phi = lambda k: np.sqrt(2) * np.sin((k - 0.5) * np.pi * t)
    mu_1 = lambda t: 20 / (1 + np.exp(-t))
    mu_2 = lambda t: -25 / (1 + np.exp(-t))


    mixture_k = [1/5, 1/5, 1/5, 1/5, 1/5]
    n = 1000
    X = np.zeros((n, len(t)))
    y = np.zeros(n)

    for i in range(n):
        r = np.random.rand()
        if r <= mixture_k[0]:
            ys = mu_1(t) + np.random.normal(0, 4) * phi(1) + np.random.normal(0, 8/3) * phi(2) + np.random.normal(0, 4/3) * phi(3)
            X[i] = ys
            y[i] = 0
        elif r <= sum(mixture_k[:2]):
            ys = mu_1(t) + np.random.normal(0, 1) * phi(1) + np.random.normal(0, 2/3) * phi(2) + np.random.normal(0, 1/3) * phi(3)
            X[i] = ys
            y[i] = 1
        elif r <= sum(mixture_k[:3]):
            ys = mu_2(t) + np.random.normal(0, 4) * phi(1) + np.random.normal(0, 8/3) * phi(2) + np.random.normal(0, 4/3) * phi(3)
            X[i] = ys
            y[i] = 2
        elif r <= sum(mixture_k[:4]):
            ys = mu_2(t) + np.random.normal(0, 1) * phi(1) + np.random.normal(0, 2/3) * phi(2) + np.random.normal(0, 1/3) * phi(3)
            X[i] = ys
            y[i] = 3
        else:
            ys = mu_2(t) + np.random.normal(0, 1) * phi(1) + np.random.normal(0, 2/3) * phi(2) + np.random.normal(0, 1/3) * phi(3) - 15 * t
            X[i] = ys
            y[i] = 4
    return skfda.FDataGrid(X, grid_points= t), y

################################### Simulation scenario B ###################################
def h_1_b(t):
    val = 6 - np.abs(t - 7)
    val[val < 0] = 0
    return val

def h_2_b(t):
    val = 6 - np.abs(t - 15)
    val[val < 0] = 0
    return val

def get_b(n = 200, n_pts = 51):
    t = np.linspace(1, 21, n_pts)
    mixture_k = [0.7, 0.3]
    X = np.zeros((n, n_pts))
    y = np.zeros(n)
    for i in range(n):
        r = np.random.rand()
        if r <= mixture_k[0]:
            X[i] = np.random.normal(0, np.sqrt(1/12)) * h_1_b(t) + np.random.normal(0, np.sqrt(1/12)) * h_2_b(t) + np.random.normal(0, np.sqrt(1/12), n_pts)
            y[i] = 0
        else:
            X[i] = np.random.normal(0, np.sqrt(1/12)) * h_1_b(t) + np.random.normal(0, np.sqrt(1/12), n_pts)
            y[i] = 1
    return skfda.FDataGrid(X, grid_points=t), y

################################### Simulation scenario C ###################################
def X_1_c(k, t):
    return -21/2 + t + k * np.random.normal(1,1) * np.cos(k * t / 10) + k * np.random.normal(1, 1) * np.sin(k + t/10) + np.random.normal(0,1, len(t))

def get_c(n = 200, n_pts = 101):
    K = 5
    t = np.linspace(1, 21, n_pts)
    X = np.zeros((n, len(t)))
    y = np.zeros(n)

    for i in range(n):
        r = np.random.rand()
        if r <= 1/K:
            X[i] = X_1_c(1, t)
            y[i] = 0
        elif r <= 2/K:
            X[i] = X_1_c(2, t)
            y[i] = 1
        elif r <= 3/K:
            X[i] = X_1_c(3, t)
            y[i] = 2
        elif r <= 4/K:
            X[i] = X_1_c(4, t)
            y[i] = 3
        else:
            X[i] = X_1_c(5, t)
            y[i] = 4
    return skfda.FDataGrid(X, grid_points=t), y

################################### Simulation scenario D ###################################
def ker_d(x, y,sig, rho):
    if x.ndim == 1:
        x = x.reshape(-1,1)
    if y.ndim == 1:
        y = y.reshape(-1,1)

    return sig * rho ** scipy.spatial.distance.cdist(x, y, 'cityblock')


def get_d(n = 200, n_pts = 101):
    t = np.linspace(0, 1, n_pts)
    kf1 = ker_d(t, t, 0.1, 0.3)
    kf2 = ker_d(t, t, 0.15, 0.35)
    kf3 = ker_d(t, t, 0.2, 0.4)

    X = np.zeros((n, n_pts))
    y = np.zeros(n)
    K = 3
    g1 = lambda t: np.exp(t) - 1
    g2 = lambda t: np.sin(np.pi * t)
    g3 = lambda t: -0.5 * t ** 2 + 0.5

    for i in range(n):
        r = np.random.rand()
        if r <= 1/K:
            X[i] = np.exp((g1(t) + np.random.uniform(0,1,n_pts) * 1 + np.random.multivariate_normal(np.zeros_like(t), kf1, 1)) / 4)
            y[i] = 0
        elif r <= 2/K:
            X[i] = np.exp((g2(t) + np.random.uniform(0,1,n_pts) * 2 + np.random.multivariate_normal(np.zeros_like(t), kf2, 1)) / 4)
            y[i] = 1
        else:
            X[i] = np.exp((g3(t) + np.random.uniform(0,1, n_pts) * 3 + np.random.multivariate_normal(np.zeros_like(t), kf3, 1)) / 4)
            y[i] = 2
    return skfda.FDataGrid(X, grid_points=t), y

################################### Simulation scenario E ###################################
def get_e(n = 200, n_pts = 101):
    x1 = lambda t: np.cos(1.5 * np.pi * t)
    x2 = lambda t: np.sin(1.5 * np.pi * t)
    x3 = lambda t: np.sin(np.pi * t)
    t = np.linspace(0, 1, n_pts)
    X = np.zeros((n, n_pts))
    y = np.zeros(n)
    K = 3

    x1t = x1(t)
    x2t = x2(t)
    x3t = x3(t)

    for i in range(n):
        r = np.random.rand()
        if r <= 1/K:
            X[i] = x1t + np.random.normal(0, 1, n_pts)
            y[i] = 0
        elif r <= 2/K:
            X[i] = x2t + np.random.normal(0, 1, n_pts)
            y[i] = 1
        else:
            X[i] = x3t + np.random.normal(0, 1, n_pts)
            y[i] = 2
    return skfda.FDataGrid(X, grid_points=t), y

################################### Simulation scenario F ###################################
def get_f():
    nb_of_samples = 30  
    t = np.expand_dims(np.linspace(0, 50, nb_of_samples), 1)
    Σ = exponentiated_quadratic(t, t) 

    mixture_k = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    n = 200
    Y = np.zeros((n, len(t)))
    simulation_label = np.zeros(n)

    c1 = lambda p: np.reshape(t, (nb_of_samples)) + np.random.multivariate_normal(
        mean=np.reshape(p * np.cos(t * p/10), (nb_of_samples)), cov=Σ, size=1) + np.random.multivariate_normal(
            mean= np.reshape(p * np.sin(p + t/10), (nb_of_samples)), cov=Σ, size=1) + np.random.normal(0, 0.125, len(t))

    c2 = lambda p: np.reshape(t, (nb_of_samples)) + np.random.multivariate_normal(mean=np.reshape(p * np.sin(t * p/10), (nb_of_samples)),
        cov=Σ, size=1) + np.random.multivariate_normal(mean=np.reshape(p * np.cos(p + t/10), (nb_of_samples)), cov = exponentiated_quadratic(5*t, 5*t),
        size=1) + np.random.multivariate_normal(mean=np.reshape(p * (np.sqrt(t/10) + t/10), (nb_of_samples)), cov=exponentiated_quadratic(8*t, 8*t),
        size=1) + np.random.normal(0, 0.125, len(t))

    for i in range(n):
        r = np.random.rand()
        if r <= mixture_k[0]:
            Y[i] = c1(1)
            simulation_label[i] = 0
        elif r <= mixture_k[0] + mixture_k[1]:
            Y[i] = c1(2)
            simulation_label[i] = 1
        elif r <= mixture_k[0] + mixture_k[1] + mixture_k[2]:
            Y[i] = c1(3)
            simulation_label[i] = 2
        elif r <= mixture_k[0] + mixture_k[1] + mixture_k[2] + mixture_k[3]:
            Y[i] = c2(1)
            simulation_label[i] = 3
        elif r <= mixture_k[0] + mixture_k[1] + mixture_k[2] + mixture_k[3] + mixture_k[4]:
            Y[i] = c2(2)
            simulation_label[i] = 4
        else:
            Y[i] = c2(3)
            simulation_label[i] = 5
    return skfda.FDataGrid(Y, grid_points=t.squeeze()), simulation_label

################################### Simulation scenario G ###################################
def get_g(n = 200, n_pts = 101):
    mu_1 = lambda t: -np.sin(t - 1) * np.log(t + 0.5)
    mu_2 = lambda t: np.cos(t) * np.log(t + 0.5)
    mu_3 = lambda t: -0.25 - 0.1 * np.cos(0.5 * (t - 1)) * t ** 1.5 * np.sqrt(5 * t ** 0.5 + 0.5)
    mu_4 = lambda t: 0.6 * np.cos(t) * np.log(t + 0.5) * np.sqrt(t + 0.5)

    def ou_ker(x, y, sig2, beta = 1):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        if y.ndim == 1:
            y = y.reshape(-1,1)
        return sig2 / (2 * beta) * np.exp(-1 * beta * scipy.spatial.distance.cdist(x, y, 'cityblock'))

    t = np.linspace(0,5, n_pts)
    X = np.zeros((n, n_pts))
    y = np.zeros(n)
    K =4

    for i in range(n):
        r = np.random.rand()
        if r <= 1/K:
            ys = np.random.multivariate_normal(mu_1(t), ou_ker(t, t, 0.1))
            X[i] = ys
            y[i] = 0
        elif r <= 2/K:
            ys = np.random.multivariate_normal(mu_2(t), ou_ker(t, t, 0.15))
            X[i] = ys
            y[i] = 1
        elif r <= 3/K:
            ys = np.random.multivariate_normal(mu_3(t), ou_ker(t, t, 0.2))
            X[i] = ys
            y[i] = 2
        else:
            ys = np.random.multivariate_normal(mu_4(t), ou_ker(t, t, 0.25))
            X[i] = ys
            y[i] = 3
    return skfda.FDataGrid(X, grid_points=t), y

################################### Simulation scenario H ###################################
def get_h():
    nb_of_samples = 30 
    t = np.expand_dims(np.linspace(0, 20, nb_of_samples), 1)
    Σ = exponentiated_quadratic(t, t) 

    mean_curve = np.zeros((3, len(t)))
    mean_curve[0] = np.reshape(np.cos(t), (nb_of_samples))
    mean_curve[1] = np.reshape(np.cos(t) -0.5*t, (nb_of_samples))
    mean_curve[2] = np.reshape(-0.5*t + 0.5, (nb_of_samples))

    mixture_k = [1/2, 1/4, 1/4]
    n = 500
    Y = np.zeros((n, len(t)))
    simulation_label = np.zeros(n)

    for i in range(n):
        r = np.random.rand()
        if r <= mixture_k[0]:
            ys = 1 + np.random.multivariate_normal(mean= mean_curve[0], cov=Σ, size=1) + np.random.normal(0, 0.2, len(t))
            Y[i] = ys
            simulation_label[i] = 1
        elif r <= mixture_k[0] + mixture_k[1]:
            ys = 2 + np.random.multivariate_normal(mean= mean_curve[1], cov=Σ, size=1) + np.random.normal(0, 0.2, len(t))
            Y[i] = ys
            simulation_label[i] = 2
        else:
            ys = np.random.multivariate_normal(mean= mean_curve[2], cov=Σ, size=1) + np.random.normal(0, 0.2, len(t))
            Y[i] = ys
            simulation_label[i] = 3
    return skfda.FDataGrid(Y, grid_points=t.squeeze()), simulation_label


################################### Simulation scenario I ###################################
def get_i():
    nb_of_samples = 30
    t = np.expand_dims(np.linspace(0, 20, nb_of_samples), 1)
    Σ = exponentiated_quadratic(t, t)

    mean_curve = np.zeros((3, len(t)))
    mean_curve[0] = np.reshape(np.cos(t), (nb_of_samples))
    mean_curve[1] = np.reshape(np.cos(t) -0.5*t, (nb_of_samples))
    mean_curve[2] = np.reshape(-0.5*t + 0.5, (nb_of_samples))

    mixture_k = [1/2, 1/4, 1/4]
    n = 500
    Y = np.zeros((n, len(t)))
    simulation_label = np.zeros(n)

    for i in range(n):
        r = np.random.rand()
        if r <= mixture_k[0]:
            ys = np.random.multivariate_normal(mean= mean_curve[0], cov=Σ, size=1)
            Y[i] = ys
            simulation_label[i] = 1
        elif r <= mixture_k[0] + mixture_k[1]:
            ys = np.random.multivariate_normal(mean= mean_curve[1], cov=Σ, size=1)
            Y[i] = ys
            simulation_label[i] = 2
        else:
            ys = np.random.multivariate_normal(mean= mean_curve[2], cov=Σ, size=1)
            Y[i] = ys
            simulation_label[i] = 3

    return skfda.FDataGrid(Y, grid_points=t.squeeze()), simulation_label


################################### Simulation scenario J ###################################
def get_j():
    nb_of_samples = 50 
    t = np.expand_dims(np.linspace(0, 10, nb_of_samples), 1)
    Σ = exponentiated_quadratic(t, t) 
    mat_ker = matern_kernel(t,t)
 
    mean_curve = np.reshape(np.cos(t), (nb_of_samples))
    mixture_k = [0.4, 0.6]
    n = 200
    Y = np.zeros((n, len(t)))
    simulation_label = np.zeros(n)

    for i in range(n):
        r = np.random.rand()
        if r <= mixture_k[0]:
            Y[i] = np.random.multivariate_normal(mean= mean_curve, cov=Σ, size=1)
            simulation_label[i] = 1
        else:
            Y[i] = np.random.multivariate_normal(mean= mean_curve, cov=mat_ker, size=1)
            simulation_label[i] = 2
    return skfda.FDataGrid(Y, grid_points=t.squeeze()), simulation_label

################################### Simulation scenario K ###################################
def get_k():
    nb_of_samples = 50 
    t = np.expand_dims(np.linspace(0, 10, nb_of_samples), 1)
    Σ = exponentiated_quadratic(t, t) 
    mat_ker = matern_kernel(t,t)

    mean_curve1 = np.reshape(np.cos(t), (nb_of_samples))
    mean_curve2 = np.reshape(np.cos(3*t), (nb_of_samples))
    mixture_k = [0.4, 0.6]
    n = 200
    Y = np.zeros((n, len(t)))
    simulation_label = np.zeros(n)

    for i in range(n):
        r = np.random.rand()
        if r <= mixture_k[0]:
            Y[i] = np.random.multivariate_normal(mean= mean_curve1, cov=Σ, size=1)
            simulation_label[i] = 1
        else:
            Y[i] = np.random.multivariate_normal(mean= mean_curve2, cov=mat_ker, size=1)
            simulation_label[i] = 2
    return skfda.FDataGrid(Y, grid_points=t.squeeze()), simulation_label


################################### Simulation scenario L ###################################
def h_1_l(t):
    val = 3 - np.abs(t - 4)
    val[val < 0] = 0
    return val

def h_2_l(t):
    val = 3 - np.abs(t - 8)
    val[val < 0] = 0
    return val

def get_l():
    nb_of_samples = 100 
    t = np.expand_dims(np.linspace(0, 10, nb_of_samples), 1)
    Σ = exponentiated_quadratic(t, t)

    mean_curve1 = h_1_l(t).squeeze()
    mean_curve2 = h_2_l(t).squeeze()
    mean_curve3 = np.cos(t).squeeze()
    mixture_k = [0.4, 0.3, 0.3]
    n = 200
    Y = np.zeros((n, len(t)))
    simulation_label = np.zeros(n)

    for i in range(n):
        r = np.random.rand()
        if r <= mixture_k[0]:
            Y[i] = np.random.multivariate_normal(mean= mean_curve1, cov=Σ, size=1)
            simulation_label[i] = 1
        elif r <= sum(mixture_k[:2]):
            Y[i] = np.random.multivariate_normal(mean= mean_curve2, cov=Σ, size=1)
            simulation_label[i] = 2
        else:
            Y[i] = np.random.multivariate_normal(mean= mean_curve3, cov=Σ, size=1)
            simulation_label[i] = 3
    return skfda.FDataGrid(Y, grid_points=t.squeeze()), simulation_label
