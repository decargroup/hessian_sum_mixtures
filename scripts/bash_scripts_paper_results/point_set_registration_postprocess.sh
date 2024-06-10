#!/bin/bash

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR=${CUR_DIR}/
PROJECT_ROOT_DIR=${SCRIPT_DIR}../../
CASE_SCRIPT_DIR=${SCRIPT_DIR}../point_set_registration/
TOP_RESULT_DIR=${PROJECT_ROOT_DIR}mc_results
MIXTURE_APPROACHES=("MM" "SM" "MSM" "HSM_STD_NO_COMPLEX" )

python3 ${CASE_SCRIPT_DIR}3_monte_carlo_postprocess.py --top_result_dir ${TOP_RESULT_DIR}\
    --monte_carlo_from_dim_string "psr_{}d_{}_full"\
    --dims_list 2 3\
    --solver LM\
    --mixture_approaches ${MIXTURE_APPROACHES[@]}\
    --top_result_dir ${TOP_RESULT_DIR}\
    --read_metrics_csv  

