import argparse


def get_parser_vanilla():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dims",
        help="Amount of dimensions",
        # default=2,
        default=1,
        type=int,
    )

    parser.add_argument(
        "--num_initial_position_per_mixture",
        help="Number of initial positions to run mixture optimization from",
        # default=2,
        type=int,
        default=10,
    )

    parser.add_argument(
        "--num_mixtures",
        help="Amount of mixtures examples to run for",
        # default=2,
        type=int,
        default=10,
    )

    parser.add_argument(
        "--first_component_weight_range",
        help="Range of weights for the first component, in the format [low_bound, high_bound]",
        nargs="+",
        type=float,
        default=[0.2, 0.8],
    )

    parser.add_argument(
        "--mean_ranges",
        nargs="+",
        help="For each mixture component, a mean has to be chosen randomly. \
            Provided here in the format [low_bound_1, high_bound_1, low_bound_2, high_bound_2] etc",
        type=float,
        default=[0, 0, -2, 2],
    )

    parser.add_argument(
        "--stddev_ranges",
        nargs="+",
        help="For each mixture, a standard deviation has to be chosen randomly. \
            Provided here in the format [low_bound_1, high_bound_1, low_bound_2, high_bound_2] etc.\
                if method_to_create_smaller_components is set to multiplied, only require 2 numbers here. ",
        type=float,
        default=[0.1, 1],
    )

    parser.add_argument(
        "--method_to_create_smaller_components",
        help="Set to either. multiplied or specified. \
            For the non-dominant components, this sets whether their standard\
            deviation is a multiple of the first, or set manually. ",
        default="multiplied",
    )

    parser.add_argument(
        "--component_multiplier_ranges",
        help="Can set multipliers for standard deviations of the smaller components, \
            where they will be calculated as first stddev times corresponding multiplier.\
            Of length equal to number of components, first element set to 1 ",
        nargs="+",
        type=float,
        default=[1, 1, 2, 10],
    )

    parser.add_argument(
        "--mixture_approaches",
        help="Which mixture approaches to use. ",
        nargs="+",
        # default=["MM", "SM", "MSM", "HSM", "HSM_STD_NO_COMPLEX"],
        default=["MM", "MSM", "HSM", "HSM_STD", "HSM_STD_NO_COMPLEX"],
        # default=["HSM_STD", "HSM"],
    )

    parser.add_argument(
        "--initial_position_ranges",
        help="Initial position ranges, in the format [xmin xmax ymin ymax] etc depending on number of dimensions. ",
        nargs="+",
        type=float,
        default=[-4, 4, -4, 4],
    )

    parser.add_argument(
        "--method_initial_position_choice",
        help="How to choose initial positions. Can be random or grid",
        default="grid",
    )

    parser.add_argument(
        "--solver_type",
        help="LM or GN",
        default="LM",
    )

    parser.add_argument(
        "--max_iters",
        help="Maximum solver iterations",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--n_jobs",
        help="Number of parallel jobs to run",
        type=int,
        # default=-1,
        default=1,
    )

    parser.add_argument(
        "--step_tol", help="Optimizer step size tolerance", type=float, default=1e-8
    )
    parser.add_argument(
        "--ftol",
        help="Optimizer relative cost decrease tolerance",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--gradient_tol",
        help="Gradient infinity norm tolerance",
        type=float,
        default=1e-10,
    )

    parser.add_argument(
        "--tau",
        help="Solver tau parameter",
        type=float,
        default=1e-11,
    )

    parser.add_argument(
        "--convergence_criterion",
        help="Which convergence criterion to use. One of:'step', 'rel_cost', 'gradient'\
            which correspond to step size, relative cost decrease, and gradient infinity norm respectively",
        type=str,
        default="step",
    )

    parser.add_argument(
        "--stylesheet",
        help="Stylesheet, plots. The plotting in this script is for debugging.",
        default="./plotstylesheet.mplstyle",
    )

    parser.add_argument(
        "--top_result_dir",
        help="Top directory for all results",
        default="/home/vassili/projects/correct_sum_mixtures_/mc_results",
    )
    parser.add_argument(
        "--monte_carlo_run_id",
        help="name of monte carlo run",
        default="test",
    )
    parser.add_argument("--use_triggs_hsm", action="store_true", default=False)
    parser.add_argument("--ceres_triggs_patch", action="store_true", default=False)
    parser.add_argument("--solver", help="Solver to use, GN or LM", default="GN")
    parser.add_argument(
        "--postprocess_only",
        action="store_true",
        help="If true only do the postprocessing with existing results",
        default=False,
    )
    parser.add_argument(
        "--read_metrics_csv",
        help="For postprocessing. \
                        If set to true, simply read the metrics for each run from a big csv file. ",
        action="store_true",
        default=False,
    )

    return parser
