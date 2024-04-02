#!/bin/bash

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR=${CUR_DIR}/
PROJECT_ROOT_DIR=${SCRIPT_DIR}../../
CASE_SCRIPT_DIR=${SCRIPT_DIR}../vanilla/
TOP_RESULT_DIR=${PROJECT_ROOT_DIR}mc_results

# ONE DIMENSION 
python3 ${CASE_SCRIPT_DIR}3_plot_hessians.py --weights 0.5 0.5\
    --means 0 0 --covariances 1 2 --stylesheet ${SCRIPT_DIR}../plotstylesheet_wide.mplstyle\
    --fig_name ${PROJECT_ROOT_DIR}figs/hessians.pdf \
    --plot_bounds -4 4

