#!/bin/bash

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR=${CUR_DIR}/
PROJECT_ROOT_DIR=${SCRIPT_DIR}../../
CASE_SCRIPT_DIR=${PROJECT_ROOT_DIR}scripts/vanilla/
TOP_RESULT_DIR=${PROJECT_ROOT_DIR}mc_results

MIXTURE_APPROACHES=("MM" "SM" "MSM" "HSM_STD_NO_COMPLEX")
# CONV_CRITERIA=("step" "rel_cost" "gradient")
CONV_CRITERIA=("step")
FIRST_COMPONENT_WEIGHT_RANGE=(0.2 0.8)
MEAN_RANGES=(0 0 -2 2 -2 2 -2 2)
COMPONENT_MULTIPLIER_RANGES=(1 1 4 10 4 10 4 10)
STEP_TOL=1e-8
# NUM_INIT_POS=100
# NUM_MIX=1000
# NUM_INIT_POS=1000
# NUM_MIX=1000

NUM_INIT_POS=1000
NUM_MIX=1000


# R_MAXES=("10")
for criterion in ${CONV_CRITERIA[@]}; do
    SUFFIX="_near_${criterion}"
    STDDEV_LOW=0.4
    STDDEV_HIGH=1
    # Near case
    # ONE DIMENSION 
    python3 ${CASE_SCRIPT_DIR}1_monte_carlo.py --dims 1\
        --num_initial_position_per_mixture ${NUM_INIT_POS} --num_mixtures ${NUM_MIX}\
        --first_component_weight_range ${FIRST_COMPONENT_WEIGHT_RANGE[@]} \
        --mean_ranges ${MEAN_RANGES[@]} \
        --stddev_ranges ${STDDEV_LOW} ${STDDEV_HIGH}\
        --component_multiplier_ranges 1 1 4 10 4 10 4 10 4 10 --mixture_approaches ${MIXTURE_APPROACHES[@]}\
        --initial_position_ranges -4 4 --solver "LM" --max_iters 100\
        --method_initial_position_choice "grid"\
        --n_jobs -1 --monte_carlo_run_id vanilla_many_components_1d${SUFFIX}\
        --convergence_criterion ${criterion} --step_tol ${STEP_TOL}\
        --top_result_dir ${TOP_RESULT_DIR}

    python3 ${CASE_SCRIPT_DIR}1_monte_carlo.py --dims 2\
        --num_initial_position_per_mixture ${NUM_INIT_POS} --num_mixtures ${NUM_MIX}\
        --first_component_weight_range ${FIRST_COMPONENT_WEIGHT_RANGE[@]} \
        --mean_ranges ${MEAN_RANGES[@]}\
        --stddev_ranges ${STDDEV_LOW} ${STDDEV_HIGH}\
        --component_multiplier_ranges ${COMPONENT_MULTIPLIER_RANGES[@]} --mixture_approaches ${MIXTURE_APPROACHES[@]}\
        --initial_position_ranges -4 4 -4 4 --solver "LM" --max_iters 100\
        --method_initial_position_choice "grid"\
        --n_jobs -1 --monte_carlo_run_id vanilla_many_components_2d${SUFFIX}\
        --convergence_criterion ${criterion} --step_tol ${STEP_TOL}\
        --top_result_dir ${TOP_RESULT_DIR}


    python3 ${CASE_SCRIPT_DIR}2_monte_carlo_postprocess.py \
        --monte_carlo_from_dim_strings vanilla_many_components_{}d_near_${criterion}\
        --mc_run_names Near --dims_list 1 2 --mixture_approaches ${MIXTURE_APPROACHES[@]}\
        --top_result_dir ${TOP_RESULT_DIR}
done