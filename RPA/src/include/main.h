/**
 * @file    main.h
 * @brief   This file contains the structure definitions for SPARC and RPA
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <mpi.h>
#include <complex.h>

#include "isddft.h"

typedef struct _RPA_OBJ {
    // MPI communicators and their parallelizing parameters
    MPI_Comm qptcomm; // Cartesian topology set up on top of a kptcomm (LOCAL)
    int npqpt;          // number of processes for paral. over q-points, symmetry reduced k-points
    int qptcommIndex;
    int nqptQptcomm;
    int qptStartIndex;
    int qptEndIndex;
    MPI_Comm omegacomm; // communicator for designated omegas
    int npomega;
    int omegacommIndex;
    int nomegaOmegacomm;
    int omegaStartIndex;
    int omegaEndIndex;
    MPI_Comm nuChi0Eigscomm; // communicator for dividing trial vectors, whose amount equals to number of desired eigs of nuChi0
    int npnuChi0Neig;
    int nuChi0EigscommIndex;
    int nnuChi0Eigscomm;
    int nuChi0EigsStartIndex;
    int nuChi0EigsEndIndex;
    MPI_Comm nuChi0EigsBridgeComm; // communicator for linking ALL processors having the same rank of nuChi0EigsComm. Every nuChi0EigsComm needs
    // a complete set of eigenvalues, eigenvectors and occupations from K-S DFT calculation
    int nuChi0EigsBridgeCommIndex; // which equals to rank of the processor in nuChi0Eigscomm
    // other settings for RPA computation
    char filename[L_STRING];
    char filename_out[L_STRING]; 
    char InDensTCubFilename[L_STRING];
    char InDensUCubFilename[L_STRING];
    char InDensDCubFilename[L_STRING];
    char InOrbitalFilename[L_STRING];
    int maxitFiltering;
    int ChebDegreeRPA;
    double tol_ErpaConverge;
    // q-points, which is k-point grid without shift after symmetry reduction
    int Nqpts_sym; // amount of q-points
    double *qptWts;
    double *q1;
    double *q2;
    double *q3;
    double *qptWts_loc;
    double *q1_loc;
    double *q2_loc;
    double *q3_loc;
    // omegas, which is integral points and weights from [0, +\infty]
    int Nomega;
    int omega_start_indx;
    int omega_end_indx;
    double *omega;
    double *omega01;
    double *omegaWts;
    double *omegaWts_loc;
    // amount of eigenvalues of \nu\chi0 to be solved
    int nuChi0Neig;
    // symmetric reduced k-point grid, saved for reading orbitals
    int Nkpts_sym;
    double *kptWts;
    double *k1;
    double *k2;
    double *k3;
    int **kPqList;
} RPA_OBJ;

typedef struct _RPA_INPUT_OBJ {
    // MPI parallelizing parameters
    int npqpt;          // number of processes for paral. over k-points
    int npomega;
    int npnuChi0Neig;
    // other settings for RPA computation
    int nuChi0Neig;
    int Nomega;
    int maxitFiltering;
    int ChebDegreeRPA;
    double tol_ErpaConverge;
    char filename[L_STRING];
    char filename_out[L_STRING]; 
    char InDensTCubFilename[L_STRING];
    char InDensUCubFilename[L_STRING];
    char InDensDCubFilename[L_STRING];
    char InOrbitalFilename[L_STRING];
} RPA_INPUT_OBJ;