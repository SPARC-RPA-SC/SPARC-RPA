/**
 * @file    main.c
 * @brief   This file contains the main function for RPA
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

 /*
 Structure of RPA program
 main
 ├──initialization_RPA
 |  ├──Initialize_SPARC_before_SetComm
 |  ├──transfer_kpoints, recalculate_kpoints
 |  ├──RPA_Input_MPI_create, set_RPA_defaults, read_RPA_inputs, RPA_copy_inputs
 |  ├──set_kPq_lists, set_omegas
 |  ├──Setup_Comms_RPA
 |  ├──Initialize_SPARC_SetComm_after
 |  └──write_settings
 ├──restore_electronicGroundState
 |  ├──restore_orbitals
 |  |  ├──read_orbitals_distributed_gamma_RPA
 |  |  └──read_orbitals_distributed_kpt_RPA
 |  ├──restore_electronDensity
 |  └──restore_eigval_occ
 ├──initialize_deltaVs
 ├──test_Hx_nuChi0
 ├──cheFSI_RPA
 |  ├──(if gamma point) collect_allXorb_allLambdas_gamma
 |  ├──(if k-point) collect_allXorb_allLambdas_kpt
 |  ├──find_min_eigenvalue
 |  ├──(if gamma point) chebyshev_filtering_gamma
 |  |  └──nuChi0_mult_vectors_gamma
 |  |     ├──sternheimer_eq_gamma
 |  |     |  └──sternheimer_eq_gamma
 |  |     |     └──sternheimer_solver_gamma
 |  |     |        ├──Sternheimer_lhs
 |  |     |        ├──set_initial_guess_deltaPsis
 |  |     |        └──block_COCG
 |  |     ├──collect_deltaRho_gamma
 |  |     └──Calculate_deltaRhoPotential_gamma
 |  |        └──poissonSolver_gamma
 |  |
 |  ├──YT_multiply_Y_gamma
 |  ├──nuChi0_mult_vectors_gamma
 |  ├──project_YT_nuChi0_Y_gamma
 |  ├──generalized_eigenproblem_solver_gamma
 |  ├──subspace_rotation_unify_eigVecs_gamma
 |  ├──evaluate_cheFSI_error_gamma
 |  |
 |  ├──(if k-point) chebyshev_filtering_kpt 
 |  |  └──nuChi0_mult_vectors_kpt
 |  |     ├──sternheimer_eq_kpt
 |  |     |  └──sternheimer_eq_kpt
 |  |     |     └──sternheimer_solver_kpt
 |  |     |        ├──Sternheimer_lhs_kpt
 |  |     |        ├──set_initial_guess_deltaPsis_kpt
 |  |     |        └──solver_kpt
 |  |     ├──collect_deltaRho_kpt
 |  |     └──Calculate_deltaRhoPotential_kpt
 |  |        └──poissonSolver_kpt
 |  ├──YT_multiply_Y_kpt
 |  ├──nuChi0_mult_vectors_kpt
 |  ├──project_YT_nuChi0_Y_kpt
 |  ├──generalized_eigenproblem_solver_kpt
 |  ├──subspace_rotation_unify_eigVecs_kpt
 |  ├──evaluate_cheFSI_error_kpt
 |  |
 |  └──compute_ErpaTerm
 |
 ├──rpaIntegrationOnOmega
 └──finalization
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "main.h"
#include "initialization_RPA.h"
#include "restoreElectronicGroundState.h"
#include "prepare_PQ_operators.h"
#include "test_Hx_nuChi0_eigSolver.h"
#include "cheFSI.h"
#include "printResult.h"
#include "finalization_RPA.h"

int main(int argc, char *argv[]) {
    // set up MPI
    MPI_Init(&argc, &argv);
    // get communicator size and my rank
    MPI_Comm comm = MPI_COMM_WORLD;
    int nproc, rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    SPARC_OBJ SPARC; // save information already existing in structure SPARC
    RPA_OBJ RPA; // save new variables needed for RPA calculation

    double t1, t2;
    
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    initialize_RPA(&SPARC, &RPA, argc, argv);

    restore_electronicGroundState(&SPARC, RPA.nuChi0Eigscomm, RPA.nuChi0EigsBridgeComm, RPA.nuChi0EigscommIndex, RPA.rank0nuChi0EigscommInWorld, RPA.k1, RPA.k2, RPA.k3, RPA.kPqSymList, RPA.Nkpts_sym);

    // prepare_PQ_operators(&SPARC, &RPA);

    initialize_deltaVs(&SPARC, &RPA);

    int testFlag = 0;
    if (testFlag) {
        test_Hx_nuChi0(&SPARC, &RPA);
    }

    // int testEigFlag = 0;
    // if (testEigFlag) {
    //     test_eigSolver(&SPARC, &RPA);
    // }

    for (int qptIndex = 0; qptIndex < RPA.Nqpts_sym; qptIndex++) {
        for (int omegaIndex = 0; omegaIndex < RPA.Nomega; omegaIndex++) {
            cheFSI_RPA(&SPARC, &RPA, qptIndex, omegaIndex);
        }
    }

    print_result(&RPA, SPARC.n_atom);

    finalize_RPA(&SPARC, &RPA);

    t2 = MPI_Wtime();
    if (rank == 0) {
        printf("The program took %.3f s.\n", t2 - t1); 
    }
    // finalize MPI
    MPI_Finalize();
    return 0;
}