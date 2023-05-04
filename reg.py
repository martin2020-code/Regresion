import numpy as np


def regreg(model,p0,x, y):
    return minimize(lambda p: err_func(model, x,y,p), p0,0.1)


def minimize(func, x0: np.ndarray, delta, atol=0.01, maxiter=100):
    x = x0
    while func(x) > atol:
        dx = np.ones(x0.shape) * 0.01
        x -= delta*grad(func,x,dx)
    return x


def grad(func, x: np.ndarray, dx: np.ndarray):
    grad_ = np.zeros(x.shape)
    for i in range(grad_.size):
        xp = np.copy(x)
        xm = np.copy(x)
        xp[i] += dx[i]
        xm[i] -= dx[i]
        grad_[i] = func(xp)-func(xm)
    grad_ /= 2*dx
    return grad_


def err_func(func, x: np.ndarray, y: np.ndarray, p: np.ndarray):
    err = 0
    err += np.sum((func(x,*p)-y)**2)
    return err




model = lambda x,*p: p[0]*x*np.exp(p[1]*x)
param = np.array([0.1,0.2])
x = np.linspace(0,5)
y = model(x,*param)
pfit = regreg(model, param + np.random.randn(2), x,y)
print()