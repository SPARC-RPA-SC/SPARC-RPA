/**
 * @file    parallelization_RPA.c
 * @brief   This file contains parallelization function for RPA calculation
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
#include "parallelization_RPA.h"
#include "parallelization.h"

void Setup_Comms_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    // The Sternheimer equation will be solved in pSPARC. pRPA is the structure saving variables not in pSPARC.
    // 1. q-point communicator, with its own q-point index (coords) and weight, saved in pRPA
    // 2. omega communicator, with its own omega value and omega weight, saved in pRPA
    // 3. nuChi0Eigs communicator, distribute all trial eigenvectors of nuChi0, saved in pRPA
    // 3.3 nuChi0Eigs bridge communicator, connect all processors in different nuChi0Eigs communicator having the same index, to broadcast eigenvalues and eigenvectors from DFT
    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // 4. every nuChi0Eigs communicator replace MPI_COMM_WORLD in Setup_Comms function of SPARC, call Setup_Comms_SPARC in RPA
    // 5. spin communicator, in pSPARC
    // 6. k-point communicator, use ALL k-points, no symmetric reduction, saved in SPARC
    // 7. band communicator, in pSPARC
    // 8. domain communicator, in pSPARC
}