import numpy as np
import scipy as sp
from scipy import stats

# from scipy import stats

if __name__ == "__main__":
    N = 88
    T = 144
    # sigma = np.abs(np.eye(N) * 1 + np.random.randn(N, N) * 0.5)
    # sigma = np.abs(np.ones([N, N]) + np.random.randn(N, N) * 0.0001)
    # print(sigma.sum())
    # omega = (
    #     np.diag(np.diag(sigma)) ** 0.5
    #     @ np.linalg.inv(sigma)
    #     @ np.diag(np.diag(sigma)) ** 0.5
    # )
    omega = np.abs(np.eye(N) * 1 + np.random.randn(N, N) * 0.1)
    # omega = np.abs(np.ones([N, N]) + np.random.randn(N, N) * 0.0001)
    df = T - 1
    print(
        "logN:{}, logeigO:{}, trace:{}, N:{}".format(
            np.log(N), np.log(np.linalg.eigh(omega)[0].sum()), np.trace(omega), N
        )
    )
    u = df * (np.log(N) - np.log(np.linalg.eigh(omega)[0].sum()) + np.trace(omega) - N)
    u_prime = (1 - 1 / (6 * df - 1) * (2 * N + 1 - 2 / (N + 1))) * u
    p = N * (N + 1) / 2
    p_value = 1 - stats.chi2.cdf(u_prime, N * (N + 1) / 2)
    print(u, u_prime, p_value)

    w = (
        N
        * T
        / 2
        * (
            # 1 / N * np.trace((omega - np.eye(N)) ** 2)
            1 / N * np.trace(np.linalg.matrix_power(omega - np.eye(N), 2))
            - N / T * (1 / N * np.trace(omega)) ** 2
            + N / T
        )
    )
    p_value_w = 1 - stats.chi2.cdf(w, N * (N + 1) / 2)
    print(w, p_value_w)
