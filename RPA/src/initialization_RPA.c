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
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <assert.h>
// this is for checking existence of files
# include <unistd.h>

#include "initialization.h"
#include "tools.h"
#include "readfiles.h"

#include "main.h"
#include "initialization_RPA.h"
#include "parallelization_RPA.h"

#define N_MEMBR_RPA 17

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
    RPA_INPUT_OBJ RPA_Input;
    /* Create new MPI struct datatype RPA_INPUT_MPI (for broadcasting) */
    MPI_Datatype RPA_INPUT_MPI;
    RPA_Input_MPI_create(&RPA_INPUT_MPI);
    if (!rank) {
        set_RPA_defaults(&RPA_Input, pSPARC);
        strncpy(RPA_Input.filename, pSPARC->filename, L_STRING);
        strncpy(RPA_Input.filename_out, pSPARC->OutFilename, L_STRING);
        read_RPA_inputs(&RPA_Input); 
        MPI_Bcast(&RPA_Input, 1, RPA_INPUT_MPI, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&RPA_Input, 1, RPA_INPUT_MPI, 0, MPI_COMM_WORLD);
    }
    MPI_Type_free(&RPA_INPUT_MPI); // free the new MPI datatype
    RPA_copy_inputs(pRPA,&RPA_Input);
    // for structure RPA, allocating space for delta orbitals, delta density, delta \nu\chi\Delta V...
    // printf("%s\n", pRPA->filename_out);
    Setup_Comms_RPA(pSPARC, pRPA); // set communicator for RPA calculation, omega/k-point/q-point?/band/domain
    if (!rank) {
        write_settings(pRPA);
    }
}

void RPA_Input_MPI_create(MPI_Datatype *RPA_INPUT_MPI) {
    RPA_INPUT_OBJ rpa_input_tmp;
    MPI_Datatype RPA_TYPES[N_MEMBR_RPA] =   {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                    MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                    MPI_DOUBLE, 
                                    MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR,
                                    MPI_CHAR};
    int blens[N_MEMBR_RPA] = {1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1,
                     L_STRING, L_STRING, L_STRING, L_STRING, L_STRING,
                     L_STRING};
    // calculating offsets in an architecture independent manner
    MPI_Aint addr[N_MEMBR_RPA],disps[N_MEMBR_RPA], base;
    int i = 0;
    MPI_Get_address(&rpa_input_tmp, &base);
    // int type
    MPI_Get_address(&rpa_input_tmp.npkpt, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npband, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npomega, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npNdx, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npNdy, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npNdz, addr + i++);
    MPI_Get_address(&rpa_input_tmp.nuChi0Neig, addr + i++);
    MPI_Get_address(&rpa_input_tmp.Nomega, addr + i++);
    MPI_Get_address(&rpa_input_tmp.maxitFiltering, addr + i++);
    MPI_Get_address(&rpa_input_tmp.ChebDegreeRPA, addr + i++);
    // double type
    MPI_Get_address(&rpa_input_tmp.tol_ErpaConverge, addr + i++);
    // char[] type
    MPI_Get_address(&rpa_input_tmp.filename, addr + i++);
    MPI_Get_address(&rpa_input_tmp.filename_out, addr + i++);
    MPI_Get_address(&rpa_input_tmp.InDensTCubFilename, addr + i++);
    MPI_Get_address(&rpa_input_tmp.InDensUCubFilename, addr + i++);
    MPI_Get_address(&rpa_input_tmp.InDensDCubFilename, addr + i++);
    MPI_Get_address(&rpa_input_tmp.InOrbitalFilename, addr + i++);
    for (i = 0; i < N_MEMBR_RPA; i++) {
        disps[i] = addr[i] - base;
    }

    MPI_Type_create_struct(N_MEMBR_RPA, blens, disps, RPA_TYPES, RPA_INPUT_MPI);
    MPI_Type_commit(RPA_INPUT_MPI);
}

void set_RPA_defaults(RPA_INPUT_OBJ *pRPA_Input, SPARC_OBJ *pSPARC) {
    pRPA_Input->npkpt = -1;          
    pRPA_Input->npband = -1;         
    pRPA_Input->npomega = -1;
    pRPA_Input->npNdx = -1;          
    pRPA_Input->npNdy = -1;          
    pRPA_Input->npNdz = -1;         
    strncpy(pRPA_Input->filename, "UNDEFINED",sizeof(pRPA_Input->filename));  
    strncpy(pRPA_Input->filename_out, "UNDEFINED",sizeof(pRPA_Input->filename_out)); 
    strncpy(pRPA_Input->InDensTCubFilename, "UNDEFINED",sizeof(pRPA_Input->InDensTCubFilename));
    strncpy(pRPA_Input->InDensUCubFilename, "UNDEFINED",sizeof(pRPA_Input->InDensUCubFilename));
    strncpy(pRPA_Input->InDensDCubFilename, "UNDEFINED",sizeof(pRPA_Input->InDensDCubFilename));
    strncpy(pRPA_Input->InOrbitalFilename, "UNDEFINED",sizeof(pRPA_Input->InOrbitalFilename));
    pRPA_Input->nuChi0Neig = (pSPARC->Nstates*10 > pSPARC->Nd) ? pSPARC->Nstates*10 : pSPARC->Nd;
    pRPA_Input->Nomega = 1;
    pRPA_Input->maxitFiltering = 20;
    pRPA_Input->ChebDegreeRPA = 2;
    pRPA_Input->tol_ErpaConverge = 1e-4;
}

void read_RPA_inputs(RPA_INPUT_OBJ *pRPA_Input) {
    char input_filename[L_STRING], str[L_STRING];
    snprintf(input_filename, L_STRING, "%s.rpa", pRPA_Input->filename);
    
    FILE *input_fp = fopen(input_filename,"r");
    
    if (input_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",input_filename);
        print_usage();
        exit(EXIT_FAILURE);
    }

    while (!feof(input_fp)) {
        int count = fscanf(input_fp,"%s",str);
        if (count < 0) continue;  // for some specific cases
        
        // enable commenting with '#'
        if (str[0] == '#' || str[0] == '\n'|| strcmpi(str,"undefined") == 0) {
            fscanf(input_fp, "%*[^\n]\n"); // skip current line
            continue;
        }
        if (strcmpi(str,"NP_KPOINT_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npkpt);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_BAND_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npband);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_OMEGA_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npomega);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_DOMAIN_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npNdx);
            fscanf(input_fp,"%d", &pRPA_Input->npNdy);
            fscanf(input_fp,"%d", &pRPA_Input->npNdz);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"N_NUCHI_EIGS:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->nuChi0Neig);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"N_OMEGA:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->Nomega);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_RPA_ENERGY:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->tol_ErpaConverge);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MAXIT_FILTERING:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->maxitFiltering);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"CHEB_DEGREE_RPA:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->ChebDegreeRPA);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"INPUT_DENS_FILE:") == 0) {
            char inputDensFnames[3][L_STRING]; // at most 3 file names
            int nInputDensFname = readStringInputsFromFile(input_fp, 3, inputDensFnames);
            if (nInputDensFname == 1) {
                strncpy(pRPA_Input->InDensTCubFilename, inputDensFnames[0], L_STRING);
            } else if (nInputDensFname == 3) {
                strncpy(pRPA_Input->InDensTCubFilename, inputDensFnames[0], L_STRING);
                strncpy(pRPA_Input->InDensUCubFilename, inputDensFnames[1], L_STRING);
                strncpy(pRPA_Input->InDensDCubFilename, inputDensFnames[2], L_STRING);
            } else {
                printf(RED "[FATAL] Density file names not provided properly! (Provide 1 file w/o spin or 3 files with spin)\n" RESET);
                exit(EXIT_FAILURE);
            }

            #ifdef DEBUG
            if (nInputDensFname >= 0) {
                printf("Density file names read = ([");
                for (int i = 0; i < nInputDensFname; i++) {
                    if (i == 0) printf("%s", inputDensFnames[i]);
                    else printf(", %s", inputDensFnames[i]);
                }
                printf("], %d)\n", nInputDensFname);
            }
            printf("Total Dens file name: %s\n", pRPA_Input->InDensTCubFilename);
            printf("Dens_up file name: %s\n", pRPA_Input->InDensUCubFilename);
            printf("Dens_dw file name: %s\n", pRPA_Input->InDensDCubFilename);
            #endif
        } else if (strcmpi(str,"INPUT_ORBITAL_FILE:") == 0) {
            char inputOrbitalFnames[3][L_STRING]; // at most 3 file names
            int nInputOrbitalFname = readStringInputsFromFile(input_fp, 3, inputOrbitalFnames);
            strncpy(pRPA_Input->InOrbitalFilename, inputOrbitalFnames[0], L_STRING);
        } else {
            printf("\nCannot recognize input variable identifier: \"%s\"\n",str);
            exit(EXIT_FAILURE);
        }
    }

    fclose(input_fp);
}

void RPA_copy_inputs(RPA_OBJ *pRPA, RPA_INPUT_OBJ *pRPA_Input) {
    pRPA->npkpt = pRPA_Input->npkpt;          
    pRPA->npband = pRPA_Input->npband;         
    pRPA->npomega = pRPA_Input->npomega;        
    pRPA->npNdx = pRPA_Input->npNdx;          
    pRPA->npNdy = pRPA_Input->npNdy;          
    pRPA->npNdz = pRPA_Input->npNdz; // the validity of np settings will be checked in function Setup_Comms_RPA
    strncpy(pRPA->filename, pRPA_Input->filename,sizeof(pRPA_Input->filename)); 
    strncpy(pRPA->filename_out, pRPA_Input->filename_out,sizeof(pRPA_Input->filename_out)); 
    strncpy(pRPA->InDensTCubFilename, pRPA_Input->InDensTCubFilename,sizeof(pRPA_Input->InDensTCubFilename));
    strncpy(pRPA->InDensUCubFilename, pRPA_Input->InDensUCubFilename,sizeof(pRPA_Input->InDensUCubFilename));
    strncpy(pRPA->InDensDCubFilename, pRPA_Input->InDensDCubFilename,sizeof(pRPA_Input->InDensDCubFilename));
    strncpy(pRPA->InOrbitalFilename, pRPA_Input->InOrbitalFilename,sizeof(pRPA_Input->InOrbitalFilename));
    pRPA->nuChi0Neig = pRPA_Input->nuChi0Neig;
    pRPA->Nomega = pRPA_Input->Nomega;
    pRPA->maxitFiltering = pRPA_Input->maxitFiltering;
    pRPA->ChebDegreeRPA = pRPA_Input->ChebDegreeRPA;
    pRPA->tol_ErpaConverge = pRPA_Input->tol_ErpaConverge;
    if (!((pRPA->nuChi0Neig > 0) && (pRPA->Nomega > 0) && (pRPA->maxitFiltering > 0) && (pRPA->ChebDegreeRPA > 0))) {
        printf("\nN_NUCHI_EIGS, N_OMEGA, MAXIT_FILTERING and CHEB_DEGREE_RPA need to be positive integer.\n");
        exit(EXIT_FAILURE);
    }
}

void write_settings(RPA_OBJ *pRPA) {
    FILE *output_fp = fopen(pRPA->filename_out,"a");
    if (output_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",pRPA->filename_out);
        exit(EXIT_FAILURE);
    }
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                            RPA Settings                                   \n");
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"N_NUCHI_EIGS: %d\n",pRPA->nuChi0Neig);
    fprintf(output_fp,"N_OMEGA: %d\n",pRPA->Nomega);
    fprintf(output_fp,"TOL_RPA_ENERGY: %.3E\n",pRPA->tol_ErpaConverge);
    fprintf(output_fp,"MAXIT_FILTERING: %d\n",pRPA->maxitFiltering);
    fprintf(output_fp,"CHEB_DEGREE_RPA: %d\n",pRPA->ChebDegreeRPA);
    fprintf(output_fp,"INPUT_DENS_FILE: %s\n",pRPA->InDensTCubFilename);
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                         RPA Parallelization                               \n");
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"NP_KPOINT_PARAL_RPA: %d\n",pRPA->npkpt);
    fprintf(output_fp,"NP_BAND_PARAL_RPA: %d\n",pRPA->npband);
    fprintf(output_fp,"NP_OMEGA_PARAL_RPA: %.3E\n",pRPA->npomega);
    fprintf(output_fp,"NP_DOMAIN_PARAL_RPA: %d %d %d\n",pRPA->npNdx, pRPA->npNdy, pRPA->npNdz);
    fclose(output_fp);
}