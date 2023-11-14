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
#include <limits.h>

#include "main.h"
#include "parallelization_RPA.h"
#include "parallelization.h"

#define min(a,b) ((a)<(b)?(a):(b))

void Setup_Comms_RPA(RPA_OBJ *pRPA, int Nspin, int Nkpts, int Nstates) {
    // The Sternheimer equation will be solved in pSPARC. pRPA is the structure saving variables not in pSPARC.
    dims_divide_Eigs(pRPA->nuChi0Neig, Nspin, Nkpts, Nstates, &pRPA->npnuChi0Neig, &pRPA->npspin, &pRPA->npkpt, &pRPA->npband);
    // 1. q-point communicator, with its own q-point index (coords) and weight, saved in pRPA
    int nprocWorld, rankWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    // 3. nuChi0Eigs communicator, distribute all trial eigenvectors of nuChi0, saved in pRPA
    pRPA->npnuChi0Neig = judge_npObject(pRPA->nuChi0Neig, nprocWorld, pRPA->npnuChi0Neig);
    pRPA->nNuChi0Eigscomm = distribute_comm_load(pRPA->nuChi0Neig, pRPA->npnuChi0Neig, rankWorld, nprocWorld, &(pRPA->nuChi0EigscommIndex), &(pRPA->nuChi0EigsStartIndex), &(pRPA->nuChi0EigsEndIndex));
    int color = (pRPA->nuChi0EigscommIndex >= 0) ? pRPA->nuChi0EigscommIndex : INT_MAX;
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &pRPA->nuChi0Eigscomm);

    pRPA->rank0nuChi0EigscommInWorld = rankWorld;
    MPI_Bcast(&pRPA->rank0nuChi0EigscommInWorld, 1, MPI_INT, 0, pRPA->nuChi0Eigscomm);
    #ifdef DEBUG
    printf("I am %d in comm world, I am in %d nuChi0Eigscomm, I will handle nuChi0Eigs %d ~ %d\n", rankWorld, pRPA->nuChi0EigscommIndex, pRPA->nuChi0EigsStartIndex, pRPA->nuChi0EigsEndIndex);
    #endif
    // 3.3 nuChi0Eigs bridge communicator, connect all processors in different nuChi0Eigs communicator having the same index, to broadcast eigenvalues and eigenvectors from DFT
    int nprocNuChi0EigsComm, rankNuChi0EigsComm;
    MPI_Comm_size(pRPA->nuChi0Eigscomm, &nprocNuChi0EigsComm);
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rankNuChi0EigsComm);
    int judgeJoinCompute = (pRPA->nuChi0EigscommIndex >= 0); // if it is 0, the processor will not join computation
    color = (judgeJoinCompute > 0) ? rankNuChi0EigsComm : INT_MAX;
    MPI_Comm_split(MPI_COMM_WORLD, color, rankNuChi0EigsComm, &pRPA->nuChi0EigsBridgeComm);
    pRPA->nuChi0EigsBridgeCommIndex = (judgeJoinCompute > 0) ? rankNuChi0EigsComm : -1;
    int rankNuChi0EigsBridgeComm; 
    MPI_Comm_rank(pRPA->nuChi0EigsBridgeComm, &rankNuChi0EigsBridgeComm);
    #ifdef DEBUG
    printf("I am %d in comm world, I am in %d nuChi0EigsBridgeComm, My rank in it is %d\n", rankWorld, pRPA->nuChi0EigsBridgeCommIndex, rankNuChi0EigsBridgeComm);
    #endif
    
    // 4. every nuChi0Eigs communicator replace MPI_COMM_WORLD in Setup_Comms function of SPARC, call Setup_Comms_SPARC in RPA
    // 5. spin communicator, in pSPARC
    // 6. k-point communicator, use ALL k-points, no symmetric reduction, saved in SPARC
    // 7. band communicator, in pSPARC
    // 8. domain communicator, in pSPARC
}

void dims_divide_Eigs(int nuChi0Neig, int Nspin, int Nkpts, int Nstates, int *npnuChi0Neig, int *npspin, int *npkpt, int *npband) {
    // this function is for computing the optimal dividance on qpts, omegas and nuChi0Eigs.
}

int judge_npObject(int nObjectInTotal, int sizeFatherComm, int npInput) {
    int npOutput = npInput;
    int maxLimit = min(sizeFatherComm, nObjectInTotal);
    if (npOutput == -1) {
        npOutput = maxLimit;
    } else if (npOutput > maxLimit) {
        npOutput = maxLimit;
    }
    return npOutput;
}

int distribute_comm_load(int nObjectInTotal, int npObject, int rankFatherComm, int sizeFatherComm, int *commIndex, int *objectStartIndex, int *objectEndIndex) {
    int sizeComm = sizeFatherComm / npObject;
    if (rankFatherComm < (sizeFatherComm - sizeFatherComm % npObject))
        *commIndex = rankFatherComm / sizeComm;
    else
        *commIndex = -1;
    
    int nObjectInComm;
    if (rankFatherComm < (sizeFatherComm - sizeFatherComm % npObject))
        nObjectInComm = nObjectInTotal / npObject + (int) (*commIndex < (nObjectInTotal % npObject));
    else
        nObjectInComm = 0;

    if (*commIndex == -1) {
        *objectStartIndex = 0;
    } else if (*commIndex < (nObjectInTotal % npObject)) {
        *objectStartIndex = *commIndex * nObjectInComm;
    } else {
        *objectStartIndex = *commIndex * nObjectInComm + nObjectInTotal % npObject;
    }
    *objectEndIndex = *objectStartIndex + nObjectInComm - 1;

    return nObjectInComm;
}
