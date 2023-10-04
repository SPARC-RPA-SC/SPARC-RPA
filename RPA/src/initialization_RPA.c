/**
 * @file    initialization.c
 * @brief   This file contains the initializing function for RPA calculation
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "main.h"
#include "initialization_RPA.h"
#include "initialization.h"
#include "readfiles.h"

void initialize_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int argc, char* argv[]) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1;
    if (!rank) {
        t1 = MPI_Wtime(); 
        pSPARC->time_start = t1;
    }
    Initialize(pSPARC, argc, argv); // include cell size, lattice vectors, mesh size and k-point grid, reading ion file & pseudopotentials
    // include number of \omega needed, number of eigenvalues to be solved and required RPA energy accuracy
    // don't forget free space allocated for pSPARC!

    // for structure RPA, allocating space for delta orbitals, delta density, delta \nu\chi\Delta V...
    
    // Setup_Comms_RPA(pRPA); // set communicator for RPA calculation, omega/k-point/q-point?/band/domain

    // write_settings(&SPARC, &pRPA);
}
