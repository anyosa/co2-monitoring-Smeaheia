import numpy as np
import pandas as pd
import random
import math


def distance_grid_loc(n1: int, n2: int, sq: bool):
    n=n1*n2
    vv = np.array(range(n1))
    vv_ones = np.ones(n1)

    uu = np.array(range(n2))
    uu_ones = np.ones(n2)

    s1 = np.dot(vv.reshape(n1, 1), uu_ones.reshape(1, n2))  # H_North
    s2 = np.dot(vv_ones.reshape(n1, 1), uu.reshape(1, n2))  # s1.T H_East
    a1 = s1.reshape(n)
    a2 = s2.reshape(n)
    ones = np.ones(n)
    distance_matrix_a1 = np.dot(a1.reshape(n, 1), ones.reshape(1, n)) - np.dot(ones.reshape(n, 1), a1.reshape(1, n))
    d2_a1 = distance_matrix_a1 * distance_matrix_a1
    distance_matrix_a2 = np.dot(a2.reshape(n, 1), ones.reshape(1, n)) - np.dot(ones.reshape(n, 1), a2.reshape(1, n))
    d2_a2 = distance_matrix_a2 * distance_matrix_a2
    H = np.sqrt(d2_a1 + d2_a2)
    if{sq==True}:
        H = d2_a1 + d2_a2
    return H


def create_index(m: int, samples: int, k: int):
    index_test = []
    for i in range(samples):
        random.seed(i)
        index_test.append(random.sample(range(0,m),k))
    return index_test


def split_by_index(l: list, index_test: list):
    m = len(l)
    index_test = index_test
    index_train = [index for index in range(0,m) if index not in index_test]
    l_test = [l[index] for index in index_test]
    l_train = [l[index] for index in index_train]
    return l_train, l_test


def arguments_gp_R0(H, phi, l_train):
    R = np.exp(-phi * H)
    df_train = pd.DataFrame()
    for i in range(len(l_train)):
        df_train = df_train.append(pd.DataFrame(l_train[i].reshape(1, l_train[i].size)), ignore_index=True)

    m_train = np.array(df_train.mean(axis=0))
    cov_train = np.array(df_train.cov())
    Sigma = np.dot(np.dot(np.sqrt(np.diag(cov_train.diagonal())), R), np.sqrt(np.diag(cov_train.diagonal())))
    L = np.linalg.cholesky(Sigma)
    ld = 2 * sum(np.log(L.diagonal()))
    il = np.linalg.inv(L)
    return m_train, ld, il


def gmcov(df):
    mean_vector = np.array(df.mean(axis=0))
    covariance_matrix = np.array(df.cov())
    return (mean_vector, covariance_matrix)


def ldil(Sigma):
    L = np.linalg.cholesky(Sigma)
    ld = 2 * sum(np.log(L.diagonal()))
    il = np.linalg.inv(L)
    return ld, il


def arguments_gp_R0_G(H, phi, l_train_R0, l_train_G):
    R = np.kron(np.exp(-phi * H), np.array([[1,-0.6],[-0.6,1]]))
    df_train_R0 = ltodf(l_train_R0)
    df_train_G = ltodf(l_train_G)

    m_train_R0, cov_R0 = gmcov(df_train_R0)
    m_train_G, cov_G = gmcov(df_train_G)
    cov = np.concatenate((cov_R0.diagonal(), cov_G.diagonal()), axis=None)
    Sigma_train = np.dot(np.dot(np.sqrt(np.diag(cov)), R), np.sqrt(np.diag(cov)))
    m_train = np.concatenate((m_train_R0, m_train_G), axis=None)

    ld_train, il_train = ldil(Sigma_train)
    return m_train, ld_train, il_train


def ltodf(l: "list of arrays"):
    l_to_df = pd.DataFrame()
    for i in range(len(l)):
        l_to_df = l_to_df.append(pd.DataFrame(l[i].reshape(1, l[i].size)), ignore_index=True)
    return (l_to_df)


def probabilities_gp(p: list):
    lnp0 = p[0]
    lnp1 = p[1]
    num0 = len(str(int(math.modf(lnp0)[1])))-1
    num1 = len(str(int(math.modf(lnp1)[1])))-1
    num=min(num0,num1)
    common_base = np.exp(10)**num
    a0 = lnp0/(10**num)
    a1 = lnp1/(10**num)
    prob0 = common_base**a0/(common_base**a0 + common_base**a1)
    prob1 = common_base**a1/(common_base**a0 + common_base**a1)
    probabilities = [round(prob0, 5), round(prob1, 5)]
    return probabilities


class LogCP2:
    def __init__(self, test, mu, ld, il, prior):
        self.n = mu.shape[0]
        self.test = test
        self.mu = mu
        self.prior = prior
        self.ld = ld
        self.il = il
        self.u = np.dot(test - mu, il)

    def value(self):
        return np.log(self.prior) + (-0.5) * (np.dot(self.u, self.u.T) + self.n * np.log(2 * np.pi) + self.ld)


def classif(l: list):
    if l[0] > l[1]:
        return 0
    else:
        return 1


def voi_mc(conditional_probabilities: np.array, values):
    expval_a_0 = values['v(x=0,a=0)'] * conditional_probabilities[...,0] + values['v(x=1,a=0)'] * conditional_probabilities[...,1]
    expval_a_1 = values['v(x=0,a=1)'] * conditional_probabilities[...,0] + values['v(x=1,a=1)'] * conditional_probabilities[...,1]
    pov = pd.DataFrame({'x=0': pd.Series(expval_a_0), 'x=1': pd.Series(expval_a_1)}).max(axis=1).mean()
    pv = pd.DataFrame({'x=0': pd.Series(expval_a_0), 'x=1': pd.Series(expval_a_1)}).mean(axis=0).max()
    voi = round(pov - pv, 5)
    return voi