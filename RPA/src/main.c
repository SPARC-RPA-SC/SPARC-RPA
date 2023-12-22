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
 ├──chebyshevFiltering 
 |  ├──(if PDEP method is used) find_min_eigenvalue
 |  ├──sternheimer_solver (block_cocg or gmres, in linearSolver.c)
 |  |  ├──(if Gamma point) sternheimer_solver_gamma
 |  |  |  └──Sternheimer_lhs, block_COCG
 |  |  └──(if k-point) sternheimer_solver_kpt
 |  |     └──Sternheimer_lhs_kpt, kpt_solver
 |  ├──collect_transfer_deltaRho
 |  └──Calculate_deltaRhoPotential (in linearSolver.c)
 |        ├──(if Gamma point) Calculate_deltaRhoPotential_gamma
 |        └──(if k-point) Calculate_deltaRhoPotential_kpt
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
#include "chebyshevFiltering.h"
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

    initialize_deltaVs(&SPARC, &RPA);

    int testFlag = 1;
    if (testFlag) {
        test_Hx_nuChi0(&SPARC, &RPA);
    }

    for (int qptIndex = 0; qptIndex < RPA.Nqpts_sym; qptIndex++) {
        for (int omegaIndex = 0; omegaIndex < RPA.Nomega; omegaIndex++) {
            chebyshev_filtering(&SPARC, &RPA, qptIndex, omegaIndex);
        }
    }

    print_result(&RPA);

    finalize_RPA(&SPARC, &RPA);

    t2 = MPI_Wtime();
    if (rank == 0) {
        printf("The program took %.3f s.\n", t2 - t1); 
    }
    // finalize MPI
    MPI_Finalize();
    return 0;
}