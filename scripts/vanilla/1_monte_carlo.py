import os
from pathlib import Path
from mixtures.vanilla_mixture.monte_carlo import (
    MonteCarloRunParameters,
    run_monte_carlo,
)
from mixtures.vanilla_mixture.parser_vanilla import get_parser_vanilla


def main(args):
    STATE_KEY = "x"
    mc_dir = os.path.join(args.top_result_dir, args.monte_carlo_run_id)
    Path(mc_dir).mkdir(parents=True, exist_ok=True)

    mc_params = MonteCarloRunParameters.from_args(args)
    # args.postprocess_only = True
    if not args.postprocess_only:
        run_monte_carlo(
            mc_params,
            args.mixture_approaches,
            STATE_KEY,
            args.top_result_dir,
            args.n_jobs,
        )

    csv_folder = os.path.join(mc_dir, "csv_folder")
    Path(csv_folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = get_parser_vanilla()
    args = parser.parse_args()
    main(args)
