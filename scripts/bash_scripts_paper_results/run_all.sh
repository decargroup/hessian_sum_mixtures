#!/bin/bash

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR=${CUR_DIR}/
PROJECT_ROOT_DIR=${SCRIPT_DIR}../../
TOP_RESULT_DIR=${PROJECT_ROOT_DIR}mc_results

# ./hessian_plots.sh
./vanilla_monte_carlo.sh
./vanilla_monte_carlo_postprocess.sh
# ./point_set_registration_monte_carlo.sh 
# ./point_set_registration_postprocess.sh
