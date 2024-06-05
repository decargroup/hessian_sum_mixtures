import numpy as np

"""
Show that our proposed normalization constant ensures positivity. 
"""


def proposed_norm_constant(alphas: np.ndarray):
    alpha_sum = np.sum(alphas)
    log_sum = 0.0
    for lv1 in range(alphas.shape[0]):
        log_sum = log_sum + alphas[lv1] * np.exp(alpha_sum / alphas[lv1])
    return np.log(log_sum)


def delta(fs: np.ndarray, alphas: np.ndarray):
    res_values = np.array([-np.log(alpha) + f for alpha, f in zip(alphas, fs)])
    kmax = np.argmin(res_values)

    sum_term = np.log(
        np.sum(
            np.array([alpha * np.exp(-f + fs[kmax]) for alpha, f in zip(alphas, fs)])
        )
    )
    neg_lse = fs[kmax] - sum_term

    coeffs = []
    for alpha, f in zip(alphas, fs):
        denominator = np.sum(
            np.array([alpha_ * np.exp(-f_ + f) for alpha_, f_ in zip(alphas, fs)])
        )
        coeffs.append(alpha / denominator)

    e_solver = np.sum(coeffs * fs)
    return neg_lse - e_solver


def main():

    np.random.seed(0)
    for num_components in range(1, 10):
        normalization_constant = lambda alphas: proposed_norm_constant(alphas)
        success_list = []
        for lv1 in range(10000):
            success = False
            alphas = np.abs(np.random.rand(num_components)) * 100
            fs = np.abs(np.random.rand(num_components)) * 1000000
            if normalization_constant(alphas) + delta(fs, alphas) > 0:
                success = True
            success_list.append(success)

        print(
            "Num components",
            num_components,
            "Succeeded",
            sum(success_list),
            "of",
            len(success_list),
        )


if __name__ == "__main__":
    main()
