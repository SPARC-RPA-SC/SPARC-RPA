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
#include "initialization.h"
#include "initialization_RPA.h"
#include "initialization_SPARC.h"
#include "parallelization_RPA.h"
#include "parallelization_SPARC.h"
#include "generateKgrid.h"

#define N_MEMBR_RPA 21

void initialize_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int argc, char* argv[]) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1;
    if (!rank) {
        t1 = MPI_Wtime(); 
        pSPARC->time_start = t1;
    }
    Initialize_SPARC_before_SetComm(pSPARC, argc, argv); // include cell size, lattice vectors, mesh size and k-point grid, reading ion file & pseudopotentials
    pRPA->deltaRhos = NULL;
    pRPA->deltaVs = NULL;
    pRPA->initDeltaVs = NULL;
    pRPA->deltaPsis = NULL;
    pRPA->deltaRhos_kpt = NULL;
    pRPA->deltaVs_kpt = NULL;
    pRPA->initDeltaVs_kpt = NULL;
    pRPA->deltaPsis_kpt = NULL;
    pRPA->Nkpts_sym = pSPARC->Nkpts_sym;
    pRPA->kptWts = (double *)malloc(pRPA->Nkpts_sym * sizeof(double));
    pRPA->k1 = (double *)malloc(pRPA->Nkpts_sym * sizeof(double));
    pRPA->k2 = (double *)malloc(pRPA->Nkpts_sym * sizeof(double));
    pRPA->k3 = (double *)malloc(pRPA->Nkpts_sym * sizeof(double));
    transfer_kpoints(pSPARC, pRPA); // pRPA saves symmetric k-point
    recalculate_kpoints(pSPARC); // pSPARC saves k-point complete grid, no symmetry. pSPARC->Nkpts_sym = pSPARC->Nkpts
    RPA_INPUT_OBJ RPA_Input;
    /* Create new MPI struct datatype RPA_INPUT_MPI (for broadcasting) */
    MPI_Datatype RPA_INPUT_MPI;
    RPA_Input_MPI_create(&RPA_INPUT_MPI);
    if (!rank) {
        set_RPA_defaults(&RPA_Input, pSPARC->Nstates, pSPARC->Nd);
        strncpy(RPA_Input.filename, pSPARC->filename, L_STRING);
        strncpy(RPA_Input.filename_out, pSPARC->OutFilename, L_STRING);
        read_RPA_inputs(&RPA_Input); 
        MPI_Bcast(&RPA_Input, 1, RPA_INPUT_MPI, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&RPA_Input, 1, RPA_INPUT_MPI, 0, MPI_COMM_WORLD);
    }
    MPI_Type_free(&RPA_INPUT_MPI); // free the new MPI datatype
    RPA_copy_inputs(pRPA,&RPA_Input);
    // set q-point grid
    pRPA->Nqpts_sym = pSPARC->Kx*pSPARC->Ky*pSPARC->Kz,
    pRPA->qptWts = (double *)malloc(pRPA->Nqpts_sym * sizeof(double));
    pRPA->q1 = (double *)malloc(pRPA->Nqpts_sym * sizeof(double));
    pRPA->q2 = (double *)malloc(pRPA->Nqpts_sym * sizeof(double));
    pRPA->q3 = (double *)malloc(pRPA->Nqpts_sym * sizeof(double));
    pRPA->Nqpts_sym = set_qpoints(pRPA->qptWts, pRPA->q1, pRPA->q2, pRPA->q3, pSPARC->Kx, pSPARC->Ky, pSPARC->Kz, pSPARC->range_x, pSPARC->range_y, pSPARC->range_z);
    pRPA->kPqList = (int **)malloc(pSPARC->Nkpts_sym * sizeof(int*)); // pSPARC saves k-point complete grid, no symmetry. pSPARC->Nkpts_sym = pSPARC->Nkpts
    for (int nk = 0; nk < pSPARC->Nkpts_sym; nk++) {
        pRPA->kPqList[nk] = (int*)malloc((pRPA->Nqpts_sym + 1) * sizeof(int));
    }
    set_kPq_lists(pRPA->Nkpts_sym, pRPA->k1, pRPA->k2, pRPA->k3, pSPARC->Nkpts_sym, pSPARC->k1, pSPARC->k2, pSPARC->k3, 
        pRPA->Nqpts_sym, pRPA->q1, pRPA->q2, pRPA->q3, pSPARC->range_x, pSPARC->range_y, pSPARC->range_z, pRPA->kPqList);
    // set integration point omegas
    pRPA->omega = (double *)malloc(pRPA->Nomega * sizeof(double));
    pRPA->omega01 = (double *)malloc(pRPA->Nomega * sizeof(double));
    pRPA->omegaWts = (double *)malloc(pRPA->Nomega * sizeof(double));
    set_omegas(pRPA->omega, pRPA->omega01, pRPA->omegaWts, pRPA->Nomega);
    
    // for structure RPA, allocating space for delta orbitals, delta density, delta \nu\chi\Delta V...
    // printf("%s\n", pRPA->filename_out);
    Setup_Comms_RPA(pRPA, pSPARC->Nspin, pSPARC->Nkpts, pSPARC->Nstates); // set communicator for RPA calculation
    pSPARC->npspin = pRPA->npspin;
    pSPARC->npkpt = pRPA->npkpt;
    pSPARC->npband = pRPA->npband;
    pSPARC->npNdx = pRPA->npNdx;
    pSPARC->npNdy = pRPA->npNdy;
    pSPARC->npNdz = pRPA->npNdz;
    pSPARC->npNdx_phi = pRPA->npNdx_phi;
    pSPARC->npNdy_phi = pRPA->npNdy_phi;
    pSPARC->npNdz_phi = pRPA->npNdz_phi;
    Initialize_SPARC_SetComm_after(pSPARC, pRPA->nuChi0Eigscomm, pRPA->nuChi0EigscommIndex, pRPA->rank0nuChi0EigscommInWorld);
    pRPA->npspin = pSPARC->npspin;
    pRPA->npkpt = pSPARC->npkpt;
    pRPA->npband = pSPARC->npband;
    pRPA->npNdx = pSPARC->npNdx;
    pRPA->npNdy = pSPARC->npNdy;
    pRPA->npNdz = pSPARC->npNdz;
    pRPA->npNdx_phi = pSPARC->npNdx_phi;
    pRPA->npNdy_phi = pSPARC->npNdy_phi;
    pRPA->npNdz_phi = pSPARC->npNdz_phi;
    MPI_Barrier(MPI_COMM_WORLD); // code above are okay
    if (!rank) {
        write_output_init(pSPARC);
        write_settings(pRPA);
    }
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        if (pSPARC->isGammaPoint) {
            pRPA->initDeltaVs = (double*)calloc(sizeof(double), pSPARC->Nd_d);
        } else {
            pRPA->initDeltaVs_kpt = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nd_d);
        }
    }
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    if (!flagNoDmcomm) {
        if (pSPARC->isGammaPoint) {
            pRPA->deltaRhos = (double*)calloc(sizeof(double), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
            pRPA->deltaVs = (double*)calloc(sizeof(double), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
            pRPA->deltaPsis = (double*)calloc(sizeof(double), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
        } else {
            pRPA->deltaRhos_kpt = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
            pRPA->deltaVs_kpt = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
            pRPA->deltaPsis_kpt = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nd_d_dmcomm * 2);
        }
    }
}

void RPA_Input_MPI_create(MPI_Datatype *RPA_INPUT_MPI) {
    RPA_INPUT_OBJ rpa_input_tmp;
    MPI_Datatype RPA_TYPES[N_MEMBR_RPA] =   {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                    MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                    MPI_INT, MPI_INT, MPI_INT, MPI_INT, 
                                    MPI_DOUBLE, 
                                    MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR,
                                    MPI_CHAR};
    int blens[N_MEMBR_RPA] = {1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1,
                     1,
                     L_STRING, L_STRING, L_STRING, L_STRING, L_STRING,
                     L_STRING};
    // calculating offsets in an architecture independent manner
    MPI_Aint addr[N_MEMBR_RPA],disps[N_MEMBR_RPA], base;
    int i = 0;
    MPI_Get_address(&rpa_input_tmp, &base);
    // int type
    MPI_Get_address(&rpa_input_tmp.npnuChi0Neig, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npspin, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npkpt, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npband, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npNdx, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npNdy, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npNdz, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npNdx_phi, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npNdy_phi, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npNdz_phi, addr + i++);
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

void set_RPA_defaults(RPA_INPUT_OBJ *pRPA_Input, int Nstates, int Nd) {
    pRPA_Input->npnuChi0Neig = -1; 
    pRPA_Input->npspin = 0;
    pRPA_Input->npkpt = 0;
    pRPA_Input->npband = 0;
    pRPA_Input->npNdx = 0;
    pRPA_Input->npNdy = 0;
    pRPA_Input->npNdz = 0;
    pRPA_Input->npNdx_phi = 0;
    pRPA_Input->npNdy_phi = 0;
    pRPA_Input->npNdz_phi = 0;
    strncpy(pRPA_Input->filename, "UNDEFINED",sizeof(pRPA_Input->filename));  
    strncpy(pRPA_Input->filename_out, "UNDEFINED",sizeof(pRPA_Input->filename_out)); 
    strncpy(pRPA_Input->InDensTCubFilename, "UNDEFINED",sizeof(pRPA_Input->InDensTCubFilename));
    strncpy(pRPA_Input->InDensUCubFilename, "UNDEFINED",sizeof(pRPA_Input->InDensUCubFilename));
    strncpy(pRPA_Input->InDensDCubFilename, "UNDEFINED",sizeof(pRPA_Input->InDensDCubFilename));
    strncpy(pRPA_Input->InOrbitalFilename, "UNDEFINED",sizeof(pRPA_Input->InOrbitalFilename));
    pRPA_Input->nuChi0Neig = (Nstates*10 > Nd) ? Nstates*10 : Nd;
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
        if (strcmpi(str,"NP_NUCHI_EIGS_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npnuChi0Neig);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_SPIN_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npspin);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_KPOINT_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npkpt);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_BAND_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npband);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_DOMAIN_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npNdx);
            fscanf(input_fp,"%d", &pRPA_Input->npNdy);
            fscanf(input_fp,"%d", &pRPA_Input->npNdz);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_DOMAIN_PHI_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npNdx_phi);
            fscanf(input_fp,"%d", &pRPA_Input->npNdy_phi);
            fscanf(input_fp,"%d", &pRPA_Input->npNdz_phi);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"N_NUCHI_EIGS:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->nuChi0Neig);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"N_OMEGA:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->Nomega);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_RPA_ENERGY:") == 0) {
            fscanf(input_fp,"%lf",&pRPA_Input->tol_ErpaConverge);
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
    pRPA->npnuChi0Neig = pRPA_Input->npnuChi0Neig; // the validity of np settings will be checked in function Setup_Comms_RPA
    pRPA->npspin = pRPA_Input->npspin;
    pRPA->npkpt = pRPA_Input->npkpt;
    pRPA->npband = pRPA_Input->npband;
    pRPA->npNdx = pRPA_Input->npNdx;
    pRPA->npNdy = pRPA_Input->npNdy;
    pRPA->npNdz = pRPA_Input->npNdz;
    pRPA->npNdx_phi = pRPA_Input->npNdx_phi;
    pRPA->npNdy_phi = pRPA_Input->npNdy_phi;
    pRPA->npNdz_phi = pRPA_Input->npNdz_phi;
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

void set_omegas(double *omega, double *omega01, double *omegaWts, int Nomega) {
    // from ABINIT, Gauss Legendre coefficient
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double tolerance = 1e-13;
    double length = (1.0 - 0.0) / 2;
    double mean = (1.0 + 0.0) / 2;
    double z;
    int flag;
    double p1, p2, p3, pp, z1;
    for (int index = 1; index < Nomega + 1; index++) {
        z = cos(M_PI*((double)index - 0.25) / ((double)Nomega + 0.5));
        flag = 1;
        while (flag) {
            p1 = 1.0;
            p2 = 0.0;
            for (int j = 1; j < Nomega + 1; j++) {
                p3 = p2;
                p2 = p1;
                p1 = ((2.0*(double)j - 1.0)*z*p2 - ((double)j - 1.0)*p3) / (double)j;
            }
            pp = (double)Nomega*(p2 - z*p1) / (1.0 - z*z);
            z1 = z;
            z = z1 - p1/pp;
            flag = (fabs(z - z1) > tolerance);
        }
        omega01[index - 1] = mean - length*z;
        omega01[Nomega - index] = mean + length*z;
        omegaWts[index - 1] = 2.0*length / ((1.0 - z*z) * (pp*pp));
        omegaWts[Nomega - index] = omegaWts[index - 1];
    }
    for (int index = 0; index < Nomega; index++) {
        omega[index] = 1.0 / omega01[index] - 1.0;
    }
    #ifdef DEBUG
    if (!rank) {
        printf("integration point omega and weight are\n");
        for (int index = 0; index < Nomega; index++)
            printf("omega %9.6f, omega01 %9.6f, weight %9.6f\n", omega[index], omega01[index], omegaWts[index]);
    }
    #endif
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
    fprintf(output_fp,"NP_NUCHI_EIGS_PARAL_RPA: %d\n",pRPA->npnuChi0Neig);
    fprintf(output_fp,"NP_SPIN_PARAL_RPA: %d\n",pRPA->npspin);
    fprintf(output_fp,"NP_KPOINT_PARAL_RPA: %d\n",pRPA->npkpt);
    fprintf(output_fp,"NP_BAND_PARAL_RPA: %d\n",pRPA->npband);
    fprintf(output_fp,"NP_DOMAIN_PARAL_RPA: %d %d %d\n",pRPA->npNdx,pRPA->npNdy,pRPA->npNdz);
    fprintf(output_fp,"NP_DOMAIN_PHI_PARAL_RPA: %d %d %d\n",pRPA->npNdx_phi,pRPA->npNdy_phi,pRPA->npNdz_phi);
    fclose(output_fp);
}