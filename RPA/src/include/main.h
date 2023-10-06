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
    MPI_Comm kptcomm;   // communicator for k-point calculations (LOCAL)
    int npkpt;          // number of processes for paral. over k-points
    MPI_Comm kptcomm_topo; // Cartesian topology set up on top of a kptcomm (LOCAL)
    MPI_Comm kptcomm_topo_excl; // processors excluded from the Cart topo within a kptcomm (LOCAL)
    MPI_Comm kptcomm_inter; // inter-communicator connecting the Cart topology and the rest in a kptcomm (LOCAL)
    MPI_Comm kpt_bridge_comm; // bridging communicator that connects all processes in kptcomm that have the same rank (LOCAL)
    MPI_Comm bandcomm;  // communicator for band calculations (LOCAL)
    int npband;         // number of processes for paral. over bands
    MPI_Comm omegacomm; // communicator for solving Sternheimer equations with designated omegas
    int npomega;
    MPI_Comm dmcomm;    // communicator for domain decomposition (LOCAL)
    int npNdx;          // number of processes for paral. over domain in x-dir
    int npNdy;          // number of processes for paral. over domain in y-dir
    int npNdz;          // number of processes for paral. over domain in z-dir
    MPI_Comm blacscomm; // communicator for using blacs to do calculation (LOCAL)
    // other settings for RPA computation
    char filename[L_STRING];
    char filename_out[L_STRING]; 
    char InDensTCubFilename[L_STRING];
    char InDensUCubFilename[L_STRING];
    char InDensDCubFilename[L_STRING];
    char InOrbitalFilename[L_STRING];
    int nuChi0Neig;
    int Nomega;
    int maxitFiltering;
    int ChebDegreeRPA;
    double tol_ErpaConverge;
} RPA_OBJ;

typedef struct _RPA_INPUT_OBJ {
    // MPI parallelizing parameters
    int npkpt;          // number of processes for paral. over k-points
    int npband;         // number of processes for paral. over bands
    int npomega;
    int npNdx;          // number of processes for paral. over domain in x-dir
    int npNdy;          // number of processes for paral. over domain in y-dir
    int npNdz;          // number of processes for paral. over domain in z-dir
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