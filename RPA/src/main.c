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
 |  ├──readInput
 |  ├──generateReducedKgrid
 |  ├──parallelization
 |  └──omegaMeshWeight
 ├──read_scf
 ├──prepare_Hamiltonian
 ├──chebyshevFiltering 
 |  ├──(if PDEP method is used) 
 |  |  nuChi0MultiplyDeltaV 
 |  |  ├──sternheimerSolver (block_cocg or gmres, in linearSolver.c)
 |  |  |  ├──(if Gamma point) blockCocg
 |  |  |  └──(if k-point) modGmres
 |  |  ├──composeDeltaRho 
 |  |  └──AARSolver (in linearSolver.c)
 |  └──rpaIntegrationOnOmega
 └──finalization
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "main.h"
#include "initialization_RPA.h"
#include "readScf.h"
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

    read_scf(&SPARC);

    // prepare_Hamiltonian(&SPARC, &RPA); // for generating Veff and Vnl by SPARC, then transferring to RPA

    chebyshev_filtering(&RPA);

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