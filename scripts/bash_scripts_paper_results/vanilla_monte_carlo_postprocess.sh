#!/bin/bash

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR=${CUR_DIR}/
PROJECT_ROOT_DIR=${SCRIPT_DIR}../../
CASE_SCRIPT_DIR=${SCRIPT_DIR}../vanilla/
TOP_RESULT_DIR=${PROJECT_ROOT_DIR}mc_results


python3 ${CASE_SCRIPT_DIR}2_monte_carlo_postprocess.py \
    --monte_carlo_from_dim_strings vanilla_many_components_{}d_near_step\
    --mc_run_names Near --dims_list 1 2 --mixture_approaches "MM" "SM" "MSM" "HSM_STD_NO_COMPLEX"\
    --top_result_dir ${TOP_RESULT_DIR}\
    --read_metrics_csv # if already ran once, csv is saved that can be loaded for metrics
