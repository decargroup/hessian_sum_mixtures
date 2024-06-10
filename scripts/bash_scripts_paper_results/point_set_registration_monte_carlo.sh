#!/bin/bash

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_DIR=${CUR_DIR}/
PROJECT_ROOT_DIR=${SCRIPT_DIR}../../
CASE_SCRIPT_DIR=${SCRIPT_DIR}../point_set_registration/
TOP_RESULT_DIR=${PROJECT_ROOT_DIR}mc_results
# Use this for fewer runs that will run faster for debuggign
# NUM_CONFIGURATIONS=4
# RUNS_PER_CONFIGURATION=4
# SUFFIX="_"

# Big run for paper
NUM_CONFIGURATIONS=100
RUNS_PER_CONFIGURATION=100
# NUM_CONFIGURATIONS=1
# RUNS_PER_CONFIGURATION=2
SUFFIX="_full"
# MIXTURE_APPROACHES=("MM" "SM" "MSM" "HSM_STD" "HSM_STD_NO_COMPLEX")
# MIXTURE_APPROACHES=("MM" "SM" "MSM" "HSM_STD" "HSM_STD_NO_COMPLEX")
MIXTURE_APPROACHES=("MM" "SM" "HSM_STD_NO_COMPLEX" "MSM")
# MIXTURE_APPROACHES=("HSM" "HSM_STD_NO_COMPLEX")
CLUSTER_FRACTION=0.3
CLUSTER_SIZE=4
STEP_TOL=1e-8
# # LM - CLUSTERED LANDMARKS
python3 ${CASE_SCRIPT_DIR}2_monte_carlo.py --dims 2\
    --num_configurations ${NUM_CONFIGURATIONS} --runs_per_configuration ${RUNS_PER_CONFIGURATION}\
    --num_landmarks 15 --fraction_of_landmarks_to_cluster ${CLUSTER_FRACTION}\
    --cluster_size ${CLUSTER_SIZE} --cluster_spread 0.1 \
    --ref_noise_stddev_bounds 0.1 0.6 --meas_noise_stddev_bounds 0.1 0.6\
    --monte_carlo_transformation_angle_range -15 15\
    --monte_carlo_transformation_r_range -0.5 0.5\
    --monte_carlo_landmark_generation_bounds -5 5\
    --top_result_dir ${TOP_RESULT_DIR} \
    --monte_carlo_run_id psr_2d_LM${SUFFIX} \
    --solver LM\
    --no_verbose\
    --n_jobs -1\
    --mixture_approaches ${MIXTURE_APPROACHES[@]}\
    --convergence_criterion step --step_tol ${STEP_TOL}\
    --no_continue_mc\
    --top_result_dir ${TOP_RESULT_DIR}\
    --stylesheet ${SCRIPT_DIR}../plotstylesheet_wide.mplstyle


# Command for 3D point set registration - basic optimization
python3 ${CASE_SCRIPT_DIR}2_monte_carlo.py --dims 3\
    --num_configurations ${NUM_CONFIGURATIONS} --runs_per_configuration ${RUNS_PER_CONFIGURATION}\
    --num_landmarks 20 --fraction_of_landmarks_to_cluster ${CLUSTER_FRACTION}\
    --cluster_size ${CLUSTER_SIZE} --cluster_spread 0.1 \
    --ref_noise_stddev_bounds 0.1 0.6 --meas_noise_stddev_bounds 0.1 0.6\
    --monte_carlo_transformation_angle_range -15 15\
    --monte_carlo_transformation_r_range -0.5 0.5\
    --monte_carlo_landmark_generation_bounds -5 5\
    --top_result_dir ${TOP_RESULT_DIR} \
    --monte_carlo_run_id psr_3d_LM${SUFFIX} \
    --solver LM\
    --no_verbose\
    --n_jobs -1\
    --mixture_approaches ${MIXTURE_APPROACHES[@]}\
    --convergence_criterion step --step_tol ${STEP_TOL}\
    --no_continue_mc\
    --top_result_dir ${TOP_RESULT_DIR}\
    --stylesheet ${SCRIPT_DIR}../plotstylesheet_wide.mplstyle


python3 ${CASE_SCRIPT_DIR}3_monte_carlo_postprocess.py --top_result_dir ${TOP_RESULT_DIR}\
    --monte_carlo_from_dim_string psr_{}d_{}${SUFFIX}\
    --dims_list 2 3\
    --solver LM\
    --mixture_approaches ${MIXTURE_APPROACHES[@]}\
    --top_result_dir ${TOP_RESULT_DIR}\
    --stylesheet ${SCRIPT_DIR}../plotstylesheet_wide.mplstyle



