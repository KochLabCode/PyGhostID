import numpy as np
import random
import matplotlib.pyplot as plt

def sort_closest(x):
    
    x_ = np.zeros(x.shape)
    x_[:,0] = x[:,0]
    for t in range(0,x.shape[1]-1):
        idcs_used = []

        for j in range(0,x.shape[0]):

            if t < 2:
                d = np.abs(x[:,t+1]-x_[j,t])
                idcs_sorted = np.argsort(d)
            else:
                    x_pred = x_[j, t] + (x_[j, t] - x_[j, t-1])
                    d = np.abs(x[:, t+1] - x_pred)
                    idcs_sorted = np.argsort(d)
            for i in range(len(idcs_sorted)):
                if idcs_sorted[i] not in idcs_used:
                    idx = idcs_sorted[i]
                    idcs_used.append(idx)
                    break
            x_[j,t+1] = x[idx,t+1]
    return x_

# from scipy.optimize import linear_sum_assignment

# def sort_assignment(x):
#     x_ = np.zeros_like(x)
#     x_[:, 0] = x[:, 0]

#     for t in range(x.shape[1] - 1):
#         # cost matrix
#         # C = np.abs(
#         #     x_[:, t][:, None] - x[:, t+1][None, :]
#         # )
#         C = (
#     np.abs(x_[:, t][:, None] - x[:, t+1][None, :]) +
#     0.1 * np.abs(
#         (x_[:, t] - x_[:, t-1])[:, None]
#         - (x[:, t+1] - x[:, t])[None, :]
#     )
# )

#         # optimal assignment
#         row_ind, col_ind = linear_sum_assignment(C)

#         x_[:, t+1] = x[col_ind, t+1]

#     return x_

# def sort_with_acceleration(x, lam=5.0):
#     # from scipy.optimize import linear_sum_assignment

#     x_ = np.zeros_like(x)
#     x_[:, :2] = x[:, :2]

#     for t in range(1, x.shape[1]-1):
#         pos = np.abs(x_[:, t][:, None] - x[:, t+1][None, :])
#         acc = np.abs(
#             (x_[:, t] - x_[:, t-1])[:, None]
#             - (x[:, t+1] - x[:, t])[None, :]
#         )
#         C = pos + lam * acc
#         _, col = linear_sum_assignment(C)
#         x_[:, t+1] = x[col, t+1]

#     return x_

# original timeseries
t = np.linspace(0,10,100)
a = np.asarray([np.sin(t), np.cos(t)+0.2, -1+0.2*t])

#shuffeled timeseries
a_ = a.copy()
for i in range(a.shape[1]):
    idcs = random.sample(range(0,3),3)
    a_[:,i]=a[idcs,i]

# re-sorted timeseries
# b = sort_with_acceleration(a_)
b = sort_closest(a_)

plt.figure()

plt.subplot(3,1,1)
plt.plot(t,a[0,:],'o-')
plt.plot(t,a[1,:],'o-')
plt.plot(t,a[2,:],'o-')

plt.subplot(3,1,2)
plt.plot(t,a_[0,:],'o-')
plt.plot(t,a_[1,:],'o-')
plt.plot(t,a_[2,:],'o-')

plt.subplot(3,1,3)
plt.plot(t,b[0,:],'o-')
plt.plot(t,b[1,:],'o-')
plt.plot(t,b[2,:],'o-')
