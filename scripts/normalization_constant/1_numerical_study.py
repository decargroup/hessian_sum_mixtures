import numpy as np

"""
The hessian-sum-mixture uses a normalization constant to ensure positivity 
of a square root input to an error term. 
There is so far no proof of the input being positive. 
However, this is validated numerically by sampling
random values for the parameters (alphas and fs), 
for different component numbers, and verifying positivity. 
"""


def max_sum_mixture_norm_constant(alphas: np.ndarray):
    return alphas.shape[0] * np.max(alphas)


def proposed_norm_constant(alphas: np.ndarray):
    msm_constant = max_sum_mixture_norm_constant(alphas)
    log_constant = alphas.shape[0] * np.abs(np.log(msm_constant))

    return (
        max(msm_constant, log_constant) + 1
    )  # The 1 is a fudge factor for numerical round-off errors.


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
    for num_components in range(1, 40):
        normalization_constant = lambda alphas: proposed_norm_constant(alphas)
        success_list = []
        for lv1 in range(10000):
            success = False
            alphas = np.abs(np.random.rand(num_components)) * 100
            fs = np.abs(np.random.rand(num_components)) * 1000000
            if normalization_constant(alphas) - delta(fs, alphas) > 0:
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
