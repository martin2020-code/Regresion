import numpy as np
from scipy.linalg import norm


N = int(100)
def anzats(x, A, B): 
    return A * x * np.exp(-B*x)


def set_data(A, B, x_max, E):
    X_ = np.zeros((2,N))
    e_k = E * np.random.uniform(-1,1,N)
    X_[0] = x_max * np.random.uniform(0,1,N)
    X_[1] = A * X_[0] * np.exp(-B * X_[0]) + e_k
    return X_


def error_func(A,B,x_: np.ndarray,y_: np.ndarray):
    return np.sum((anzats(A,B,x_) - y_)**2)


def gradient(A,A_max,A_min,B,B_max,B_min,res,x_,y_):
    gradient_ = np.zeros((2,))
    gradient_[0] = (error_func(A + (A_max - A_min) / res, B, x_, y_) - error_func(A - (A_max - A_min) / res, B, x_, y_)) / (2. * (A_max - A_min) / res)
    gradient_[1] = (error_func(A,B + (B_max - B_min) / res, x_, y_) - error_func(A, B - (B_max - B_min) / res, x_, y_)) / (2. * (B_max - B_min) / res)
    return gradient_
    
    
# Set the data
A_ = 1.24; B_ = 0.87; E = 0.05; x_max = 5.

x_, y_ = set_data(A_,B_,x_max,E)


# Save the data in 'scatterred.dat'
dswk = open('pyreg/scattered.dat', 'w+')
for k in range(N):
    dswk.write(f'{x_[k]:g} {y_[k]:g}\n')
dswk.close()


# Save error parameter surface
A_min = 0.; A_max = 2.; B_min = 0.; B_max = 2.; res = 200.
dswk = open('pyreg/surface.dat', 'w+')
A = A_min
while A < A_max:
    B = B_min
    while B < B_max:
        dswk.write(f'{A:g} {B:g} {error_func(A,B,x_,y_):g}\n')
        B += (B_max - B_min) / res
    dswk.write('\n')
    A += (A_max - A_min) / res
dswk.close()

# Calculate gradient vector field
res = 50
dswk = open('pyreg/gradient.dat', 'w+')
A_min = 0.
A_max = 2.
B_min = 0.
B_max = 2.

A = A_min
while A < A_max:
    B = B_min
    while B < B_max:
        gradient_ = gradient(A, A_max, A_min, B, B_max, B_min, res, x_, y_)
        DA = -gradient_[0]
        DB = -gradient_[1]
        theta = np.arctan2(DB, DA)
        p = np.array([
            A - 0.5 * ((A_max - A_min) / res) * np.cos(theta),
            B - 0.5 * ((B_max - B_min) / res) * np.sin(theta),
            ((A_max - A_min) / res) * np.cos(theta),
            ((B_max - B_min) / res) * np.sin(theta),
            np.log(np.sqrt(DA**2 + DB**2))
        ])
        dswk.write(f'{p[0]:g} {p[1]:g} {p[2]:g} {p[3]:g} {p[4]:g}')
        B += (B_max - B_min) / res
    A += (A_max - A_min) / res
dswk.close()


# Gradient descent
dswk = open('pyreg/delta_function.dat', 'w+')
dswk2 = open('pyreg/minima.dat', 'w+')

for i in range(100):
    Ak = (np.random.uniform(0,1)) * (A_max - A_min) + A_min
    Bk = (np.random.uniform(0,1)) * (B_max - B_min) + B_min
    Ak_ = Ak
    Bk_ = Bk
    
    while True:
        Ak =Ak_
        Bk = Bk_
        gradientk = gradient(Ak, A_max, A_min, Bk, B_max, B_min, res, x_, y_)
        theta = np.arctan2(-gradientk[1], -gradientk[0])
        dot = 1.
        delta = 0
        while True:
            delta += 0.0001
            Ak_ = Ak + delta * np.cos(theta);
            Bk_ = Bk + delta * np.sin(theta);
            
            gradientk_ = gradient(Ak_, A_max, A_min, Bk_, B_max, B_min, res, x_, y_)
            dot = np.dot(gradientk, gradientk_) / (np.dot(gradientk, gradientk) * np.dot(gradientk_, gradientk_))
            
            if i < 8:
                p = np.array([
                    Ak_, Bk_,
                    error_func(Ak_, Bk_, x_, y_),
                    delta, i+1
                ])
                dswk.write(f'{p[0]:g} {p[1]:g} {p[2]:g} {p[3]:g} {int(p[4]):d}\n')
        

            if dot > 0.:
                break
        
        if not np.sqrt((Ak - Ak_)**2 + (Bk - Bk_)**2) > 1e-4:
            break
        
    dswk2.write(f'{Ak_:g} {Bk_:g} {error_func(Ak_,Bk_,x_,y_):g}\n')

dswk.close()
dswk2.close()
