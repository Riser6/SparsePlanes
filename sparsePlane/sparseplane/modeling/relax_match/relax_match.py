import torch
import torch.nn.functional as F
import numpy as np
import time 
import logging
from scipy.optimize import linear_sum_assignment


def project_row(X):
    """
    p(X) = X - 1/m (X 1m - 1n) 1m^T
    X shape: n x m
    """

    X_row_sum = X.sum(dim=1, keepdim=True) # shape n x 1
    one_m = torch.ones(1, X.shape[1]).to(X.device) # shape 1 x m

    mask = (X_row_sum <= 1).float()
    P = X - (X_row_sum - 1).mm(one_m) / X.shape[1]

    return X * mask + (1-mask) * P

    # return X - (X_row_sum - 1).mm(one_m) / X.shape[1]

def project_col(X):
    """
    p(X) = X                                                        if X^T 1n <= 1m
    p(X) = X - 1/n 1n (1n^T X - 1m^T)       else
    X shape: n x m
    """
    X_col_sum = X.sum(dim=0, keepdim=True) # shape 1 x m
    one_n = torch.ones(X.shape[0], 1).to(X.device) # shape n x 1

    mask = (X_col_sum <= 1).float()
    P = X - (one_n).mm(X_col_sum - 1) / X.shape[0]

    return X * mask + (1 - mask) * P        


def relax_matching(C_batched, numPlanes1, numPlanes2, max_iter=100, proj_iter=100, lr=0.1, return_time=0):
    """
    C shape: n: number of template, m: number of proposals
    """
    X_all = torch.zeros_like(C_batched)
    for idx, (C_padded, numP1, numP2) in enumerate(zip(C_batched, numPlanes1, numPlanes2)):
        # ----------------------------------------------------
        # row min init: 
        # ----------------------------------------------------
        numP1 = numP1[0]
        numP2 = numP2[0]
        C = C_padded[:numP1, :numP2]
        X = torch.zeros_like(C)
        C_rowmin = C.clone() #.numpy()
        for m in range(C_rowmin.shape[1]):
            largest_ind = torch.argmin(C_rowmin[:, m])
            for n in range(C_rowmin.shape[0]):
                if n != largest_ind: 
                    C_rowmin[n,m] = C.max() 
        # C_rowmin = torch.from_numpy(C_rowmin)
        _, idx_max = torch.min(C_rowmin, dim=1)
        assert(idx_max.shape[0] == C.shape[0])
        X[torch.arange(C.shape[0]).long(), idx_max.long()] = 1.0

        # project C onto the constrain set 
        P = [torch.zeros_like(C) for _ in range(3)] 
        cost = [0]
        stime = time.time()    
        for i in range(max_iter):
            X = X - lr * C # gradient step
            cost.append((X * C).norm().item())
            for j in range(proj_iter):
                X_start = X.clone()       
                X = X + P[0]
                Y = F.relu(X)
                P[0] = X - Y

                X = Y + P[1]
                Y = project_col(X)
                P[1] = X - Y

                X = Y + P[2]
                Y = project_row(X)
                P[2] = X - Y

                X = Y
                if (X - X_start).norm().item() == 0:
                    break
            #print('iter %d(%d) D: proj_error: %.8f| cost %.8f '%(i, j, inner_projection_error[-1], cost[-1]))
            #print('iter %d(%d) P: %.4f %.4f %.4f'%(i, j, P[0].norm().item(), P[1].norm().item(),
            #    P[2].norm().item()))
            if cost[-2] == cost[-1]:  # the reduced cost 
                # higher += 1
                break
        etime = time.time() - stime 
        X_all[idx][:numP1, :numP2] = F.relu(X)
    #logging.info('mtime %.3f'%etime)
    if return_time:
        return X_all, cost, etime
    else:
        return X_all



def relax_matching_origin(C, max_iter=100, proj_iter=100, lr=0.1, return_time=0):
    """
    C shape: n: number of template, m: number of proposals
    """
    n, m = C.shape        

    # ----------------------------------------------------
    # row min init: 
    # ----------------------------------------------------
    X = torch.zeros_like(C)
    C_rowmin = C.clone() #.numpy()
    for m in range(C_rowmin.shape[1]):
        largest_ind = torch.argmin(C_rowmin[:, m])
        for n in range(C_rowmin.shape[0]):
            if n != largest_ind: 
                C_rowmin[n,m] = C.max() 
    # C_rowmin = torch.from_numpy(C_rowmin)
    _, idx_max = torch.min(C_rowmin, dim=1)
    assert(idx_max.shape[0] == C.shape[0])
    X[torch.arange(C.shape[0]).long(), idx_max.long()] = 1.0

    # project C onto the constrain set 
    X_list = []
    # X = C
    X_list.append(X)

    P = [torch.zeros_like(C) for _ in range(3)] 

    inner_projection_error_list = []
    inner_projection_error = 0
    cost = [0]
    stime = time.time()    
    for i in range(max_iter):               
        X = X - lr * C # gradient step
        cost.append((X * C).norm().item())
        X_list.append(X)
        inner_projection_error = []
        for j in range(proj_iter):
            X_start = X.clone()       
            X = X + P[0]
            Y = F.relu(X)
            P[0] = X - Y

            X = Y + P[1]
            Y = project_col(X)
            P[1] = X - Y

            X = Y + P[2]
            Y = project_row(X)
            P[2] = X - Y

            X = Y
            if (X - X_start).norm().item() == 0:
                break
            #    print('early break at iter %d(%d)'%(i, j))
            inner_projection_error.append((X - X_start).norm().item())
        inner_projection_error_list.append(inner_projection_error)
        #print('iter %d(%d) D: proj_error: %.8f| cost %.8f '%(i, j, inner_projection_error[-1], cost[-1]))
        #print('iter %d(%d) P: %.4f %.4f %.4f'%(i, j, P[0].norm().item(), P[1].norm().item(),
        #    P[2].norm().item()))
        if cost[-2] == cost[-1]:  # the reduced cost 
            # higher += 1
            break
    etime = time.time() - stime 

    #logging.info('mtime %.3f'%etime)
    if return_time:
        return X, cost, X_list, inner_projection_error, etime
    else:
        return F.relu(X), cost, X_list, inner_projection_error


def test():
    # cost = np.array([[4, 1, 3], [2, 0, 5], [2, 2, 2]]) - 2
    margin = 0.2
    cost = 1 - get_cost()
    row_ind, col_ind = linear_sum_assignment(cost)
    X_0 = np.zeros_like(cost)
    X_0[row_ind, col_ind] = 1
    X_0 = torch.from_numpy(X_0).float()

    C = torch.from_numpy(cost - margin).float()
    X = relax_matching_origin(C, max_iter=100, proj_iter=100, lr=0.1)
    print('Hungarian matching = {}'.format(np.argwhere(X_0.numpy() > 0)))
    print('Relaxed matching = {}'.format(np.argwhere(X[0].numpy() > 0)))
    print('cost = {}'.format(cost))
    np.set_printoptions(precision=1,suppress=True)
    print('X = {}'.format(X[0].numpy()))
    print('X > 0.5: {}'.format(X[0] > 0.5))
    print('X_0 = {}'.format(X_0))
    print('diff = {}'.format((X[0] - X_0).norm()))

    
def hungarian_matching(cost):
    cost = cost.cpu().numpy() 
    row_ind, col_ind = linear_sum_assignment(cost)
    X_0 = np.zeros_like(cost)
    X_0[row_ind, col_ind] = 1
    X_0 = torch.from_numpy(X_0).float().cuda() 
    return X_0, None, None, None

def get_cost():
    return np.array([
        [.75,.19,.14,.25,.53,.19,.27,.98],
        [.53,.20,.72,.35,.68,.46,.63,.15],
        [.98,.23,.39,.79,.36,.14,.97,.45],
        [.14,.96,.49,.92,.57,.51,.41,.29],
        [.24,.66,.30,.19,.92,.99,.13,.15],
        [.87,.38,.71,.68,.15,.24,.95,.33],
        [.76,.71,.53,.99,.43,.46,.91,.29],
        [.74,.84,.45,.97,.18,.45,.93,.27],
        [.40,.96,.22,.78,.11,.65,.71,.24],
        [.41,.14,.48,.37,.99,.90,.27,.20],
        [.41,.46,.41,.16,.22,.34,.22,.94],
        [.57,.16,.97,.30,.43,.22,.50,.32],
        [.48,.34,.55,.59,.99,.92,.28,.15],
        [.66,.65,.64,.55,.10,.27,.61,.85],
        [.98,.22,.54,.59,.36,.31,.96,.44],
    ])

def test_given_cost(cost):
    # cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    row_ind, col_ind = linear_sum_assignment(cost)
    X_0 = np.zeros_like(cost)
    X_0[row_ind, col_ind] = 1
    X_0 = torch.from_numpy(X_0).float()

    C = torch.from_numpy(cost).float()
    X = relax_matching_origin(C, max_iter=100, proj_iter=100, lr=0.1)
    print('Hungarian matching = {}'.format(X_0))
    print('Relaxed matching = {}'.format(X))
    print('diff = {}'.format((X - X_0).norm()))


if __name__ == '__main__':
    test()