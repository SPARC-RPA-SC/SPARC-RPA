#include <complex.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "electronicGroundState.h"
#include "electrostatics.h"
#include "nlocVecRoutines.h"
#include "sqNlocVecRoutines.h"
#include "sqEnergy.h"
#include "sqProperties.h"
#include "spinOrbitCoupling.h"
#include "orbitalElecDensInit.h"
#include "orbitalElecDensInit.h"
#include "tools.h"

#include "restoreElectronicGroundState.h"

void restore_electronicGroundState(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld, double *symk1, double *symk2, double *symk3, int **kPqList) {
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    
#ifdef DEBUG
    if (rank == 0) printf("Start ground-state calculation.\n");
#endif
    
    #ifdef DEBUG
    double t1, t2;
    if (rank == 0) printf("Calculating electron density ... \n");
    t1 = MPI_Wtime();
#endif
    
    // find atoms that influence the process domain
    GetInfluencingAtoms(pSPARC);
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nFinding influencing atoms took %.3f ms\n",(t2-t1)*1000); 
    t1 = MPI_Wtime();
#endif
    
    // calculate pseudocharge density b
    Generate_PseudoChargeDensity(pSPARC);
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nCalculating b & b_ref took %.3f ms\n",(t2-t1)*1e3);
    t1 = MPI_Wtime();   
#endif

    if (pSPARC->SQFlag == 1) {
        SQ_OBJ *pSQ = pSPARC->pSQ;
        GetInfluencingAtoms_nloc(pSPARC, &pSPARC->Atom_Influence_nloc_kptcomm, 
                        pSQ->DMVertices_PR, pSQ->dmcomm_SQ);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nFinding nonlocal influencing atoms in dmcomm_SQ took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
    #endif

        CalculateNonlocalProjectors(pSPARC, &pSPARC->nlocProj_kptcomm, 
                        pSPARC->Atom_Influence_nloc_kptcomm, pSQ->DMVertices_PR, pSQ->dmcomm_SQ);

    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nCalculating nonlocal projectors in dmcomm_SQ took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
    #endif

        GetNonlocalProjectorsForNode(pSPARC, pSPARC->nlocProj_kptcomm, &pSPARC->nlocProj_SQ, 
                        pSPARC->Atom_Influence_nloc_kptcomm, &pSPARC->Atom_Influence_nloc_SQ, pSQ->dmcomm_SQ);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nGetting nonlocal projectors for each node in dmcomm_SQ took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
    #endif
        
        // TODO: Add correction term 
        if(pSPARC->SQ_correction == 1) {
            OverlapCorrection_SQ(pSPARC);
            OverlapCorrection_forces_SQ(pSPARC);
        }
    } else {
        // find atoms that have nonlocal influence the process domain (of psi-domain)
        GetInfluencingAtoms_nloc(pSPARC, &pSPARC->Atom_Influence_nloc, pSPARC->DMVertices_dmcomm, 
                                pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nFinding nonlocal influencing atoms in psi-domain took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
    #endif
        
        // calculate nonlocal projectors in psi-domain
        if (pSPARC->isGammaPoint)
            CalculateNonlocalProjectors(pSPARC, &pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                                        pSPARC->DMVertices_dmcomm, pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
        else
            CalculateNonlocalProjectors_kpt(pSPARC, &pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                                            pSPARC->DMVertices_dmcomm, pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);	                            
        
        if (pSPARC->SOC_Flag) {
            CalculateNonlocalProjectors_SOC(pSPARC, pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                                            pSPARC->DMVertices_dmcomm, pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
            CreateChiSOMatrix(pSPARC, pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                            pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
        }
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nCalculating nonlocal projectors in psi-domain took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();   
    #endif
        
        // find atoms that have nonlocal influence the process domain (of kptcomm_topo)
        GetInfluencingAtoms_nloc(pSPARC, &pSPARC->Atom_Influence_nloc_kptcomm, pSPARC->DMVertices_kptcomm, 
                                pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nFinding nonlocal influencing atoms in kptcomm_topo took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
    #endif
        
        // calculate nonlocal projectors in kptcomm_topo
        if (pSPARC->isGammaPoint)
            CalculateNonlocalProjectors(pSPARC, &pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                                        pSPARC->DMVertices_kptcomm, 
                                        pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
        else
            CalculateNonlocalProjectors_kpt(pSPARC, &pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                                            pSPARC->DMVertices_kptcomm, 
                                            pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);								    
        
        if (pSPARC->SOC_Flag) {
            CalculateNonlocalProjectors_SOC(pSPARC, pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                                            pSPARC->DMVertices_kptcomm, 
                                            pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
            CreateChiSOMatrix(pSPARC, pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                            pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
        }
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nCalculating nonlocal projectors in kptcomm_topo took %.3f ms\n",(t2-t1)*1000);   
    #endif
        
        // restore orbitals psi
        restore_orbitals(pSPARC, nuChi0Eigscomm, nuChi0EigsBridgeComm, nuChi0EigscommIndex, rank0nuChi0EigscommInWorld, symk1, symk2, symk3, kPqList);
    }

    // initialize electron density rho (initial guess)
    restore_electronDensity(pSPARC, nuChi0Eigscomm, nuChi0EigsBridgeComm, nuChi0EigscommIndex, rank0nuChi0EigscommInWorld);
    
}

void restore_orbitals(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld, double *symk1, double *symk2, double *symk3, int **kPqList) {
    if ((nuChi0EigscommIndex == -1)) return; // not take part in computation
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    int flagPrintNuChi0Eigscomm = (nuChi0EigscommIndex == 0); // set which nuchi0Eigscomm will print output
#ifdef DEBUG
    if (flagPrintNuChi0Eigscomm && (rank == 0)) printf("restoring Kohn-Sham orbitals ... \n");
#endif

    int DMnd, DMndsp, size_k, len_tot;
#ifdef DEBUG
    double t1, t2;
#endif
    // Multiply a factor for a spinor wavefunction
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    size_k = DMndsp * pSPARC->Nband_bandcomm;
    // notice that in processors not for orbital calculations len_tot = 0
    len_tot = size_k * pSPARC->Nkpts_kptcomm;

    if (pSPARC->isGammaPoint){
        // allocate memory in the very first relax/MD step
        pSPARC->Xorb = (double *)malloc( len_tot * sizeof(double) );
        pSPARC->Yorb = (double *)malloc( size_k * sizeof(double) );
        if (pSPARC->Xorb == NULL || pSPARC->Yorb == NULL) {
            printf("\nMemory allocation failed!\n");
            exit(EXIT_FAILURE);
        }
#ifdef DEBUG            
        t1 = MPI_Wtime();
#endif
        if (nuChi0EigscommIndex == 0) { // read orbitals, then broadcast through nuChi0EigsBridgeCommIndex
            read_orbitals_distributed_gamma_RPA(pSPARC, nuChi0Eigscomm); // read orbitals of kpts
            if (pSPARC->dmcomm != MPI_COMM_NULL) {
                MPI_Bcast(pSPARC->Xorb, len_tot, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
            }
        } else { // receiver of broadcast
            if (pSPARC->dmcomm != MPI_COMM_NULL) {
                MPI_Bcast(pSPARC->Xorb, len_tot, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
            }
        }
        if (pSPARC->dmcomm != MPI_COMM_NULL) {
            int XorbIndex = 2;
            printf("I am rank %d in nuChi0Eigscomm %d, pSPARC->Xorb[%d] is %.6E\n", rank, nuChi0EigscommIndex, XorbIndex, pSPARC->Xorb[XorbIndex]);
        }
    } else {
        // allocate memory in the very first relax/MD step
        pSPARC->Xorb_kpt = (double _Complex *) malloc( len_tot * sizeof(double _Complex) );
        pSPARC->Yorb_kpt = (double _Complex *) malloc( size_k * sizeof(double _Complex) );
        if (pSPARC->Xorb_kpt == NULL || pSPARC->Yorb_kpt == NULL) {
            printf("\nMemory allocation failed!\n");
            exit(EXIT_FAILURE);
        }
#ifdef DEBUG            
        t1 = MPI_Wtime();
#endif                
        if (nuChi0EigscommIndex == 0) { // read orbitals, then broadcast through nuChi0EigsBridgeCommIndex
            read_orbitals_distributed_kpt_RPA(pSPARC, nuChi0Eigscomm, symk1, symk2, symk3, kPqList); // read orbitals of kpts
            if (pSPARC->dmcomm != MPI_COMM_NULL) {
                MPI_Bcast(pSPARC->Xorb_kpt, len_tot, MPI_DOUBLE_COMPLEX, 0, nuChi0EigsBridgeComm);
            }
        } else { // receiver of broadcast
            if (pSPARC->dmcomm != MPI_COMM_NULL) {
                MPI_Bcast(pSPARC->Xorb_kpt, len_tot, MPI_DOUBLE_COMPLEX, 0, nuChi0EigsBridgeComm);
            }
        }
        if (pSPARC->dmcomm != MPI_COMM_NULL) {
            int XorbIndex = 2;
            printf("I am rank %d in nuChi0Eigscomm %d, start k %d, |%d| - 1 in sym kpt list, pSPARC->Xorb[%d] is %.6E + %.6Ei\n", rank, nuChi0EigscommIndex,
                 pSPARC->kpt_start_indx, kPqList[pSPARC->kpt_start_indx][0], XorbIndex, creal(pSPARC->Xorb_kpt[XorbIndex]),  cimag(pSPARC->Xorb_kpt[XorbIndex]));
        }
    }        
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Finished setting random orbitals. Time taken: %.3f ms\n",(t2-t1)*1e3);
#endif
}

void read_orbitals_distributed_gamma_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm) {
    if (pSPARC->dmcomm == MPI_COMM_NULL) return; // only dmcomm != NULL, the processor can join read; in this section, read only orbitals belonging to kpts inside symmetric kpt list
    int spin, band;
    int gridsizes[3]; // z, y, x
    gridsizes[0] = pSPARC->Nz; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nx;
    int localGridsizes[3]; // z, y, x
    localGridsizes[0] = pSPARC->Nz_d_dmcomm; localGridsizes[1] = pSPARC->Ny_d_dmcomm; localGridsizes[2] = pSPARC->Nx_d_dmcomm;
    int localGridStart[3]; // z, y, x
    localGridStart[0] = pSPARC->DMVertices_dmcomm[4]; localGridStart[1] = pSPARC->DMVertices_dmcomm[2]; localGridStart[2] = pSPARC->DMVertices_dmcomm[0];
    MPI_Datatype dataType = MPI_DOUBLE_COMPLEX;
    MPI_Datatype domainSubarray;
    MPI_Type_create_subarray(3, gridsizes, localGridsizes, localGridStart, MPI_ORDER_C, dataType, &domainSubarray);
    MPI_Type_commit(&domainSubarray);
    for (band = pSPARC->band_start_indx; band < pSPARC->band_end_indx + 1; band++) {
        for (spin = pSPARC->spin_start_indx; spin < pSPARC->spin_end_indx + 1; spin++) {
                read_orbitals_distributed_real_RPA(pSPARC, band, spin, domainSubarray, pSPARC->Xorb);
        }
    }
    MPI_Type_free(&domainSubarray);
}

void read_orbitals_distributed_real_RPA(SPARC_OBJ *pSPARC, int band, int spin, MPI_Datatype domainSubarray, double *readXorb) {
    MPI_Status status;
    MPI_File orbitalFh;
    char orbitalFileName[100];
    int localSpin = spin - pSPARC->spin_start_indx;
    int localBand = band - pSPARC->band_start_indx;
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    snprintf(orbitalFileName, 100, "orbitals/band%d_spin%d.orbit", band, spin);
    printf("the name of the orbital file is %s\n", orbitalFileName);
    MPI_File_open(pSPARC->dmcomm, orbitalFileName, MPI_MODE_RDWR, MPI_INFO_NULL, &orbitalFh); // cannot create a file
    MPI_File_set_view(orbitalFh, 0, MPI_DOUBLE, domainSubarray, "native", MPI_INFO_NULL);
    MPI_File_read_all(orbitalFh, readXorb + localBand*DMndsp + localSpin*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, &status);
    MPI_File_close(&orbitalFh);
}

void read_orbitals_distributed_kpt_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double *symk1, double *symk2, double *symk3, int **kPqList) {
    if (pSPARC->dmcomm != MPI_COMM_NULL) { // only dmcomm != NULL, the processor can join read; in this section, read only orbitals belonging to kpts inside symmetric kpt list
        int spin, kpt, kptSym, band;
        int gridsizes[3]; // z, y, x
        gridsizes[0] = pSPARC->Nz; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nx;
        int localGridsizes[3]; // z, y, x
        localGridsizes[0] = pSPARC->Nz_d_dmcomm; localGridsizes[1] = pSPARC->Ny_d_dmcomm; localGridsizes[2] = pSPARC->Nx_d_dmcomm;
        int localGridStart[3]; // z, y, x
        localGridStart[0] = pSPARC->DMVertices_dmcomm[4]; localGridStart[1] = pSPARC->DMVertices_dmcomm[2]; localGridStart[2] = pSPARC->DMVertices_dmcomm[0];
        MPI_Datatype dataType = MPI_DOUBLE_COMPLEX;
        MPI_Datatype domainSubarray;
        MPI_Type_create_subarray(3, gridsizes, localGridsizes, localGridStart, MPI_ORDER_C, dataType, &domainSubarray);
        MPI_Type_commit(&domainSubarray);
        for (kpt = pSPARC->kpt_start_indx; kpt < pSPARC->kpt_end_indx + 1; kpt++) {
            if (kPqList[kpt][0] < 0) continue; // this kpt is not in symmetric kpt list
            kptSym = kPqList[kpt][0] - 1;
            for (band = pSPARC->band_start_indx; band < pSPARC->band_end_indx + 1; band++) {
                for (spin = pSPARC->spin_start_indx; spin < pSPARC->spin_end_indx + 1; spin++) {
                        read_orbitals_distributed_complex_RPA(pSPARC, 1, kpt, symk1[kptSym], symk2[kptSym], symk3[kptSym], band, spin, domainSubarray, pSPARC->Xorb_kpt);
                }
            }
        }
        MPI_Type_free(&domainSubarray);
    }
    MPI_Barrier(nuChi0Eigscomm); // prevent file-read conflict between kpt comms inside sym list and outside sym list
    if (pSPARC->dmcomm != MPI_COMM_NULL) { // only dmcomm != NULL, the processor can join read; in this section, read only orbitals belonging to kpts outside symmetric kpt list
        int spin, kpt, kptSym, band;
        int gridsizes[3]; // z, y, x
        gridsizes[0] = pSPARC->Nz; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nx;
        int localGridsizes[3]; // z, y, x
        localGridsizes[0] = pSPARC->Nz_d_dmcomm; localGridsizes[1] = pSPARC->Ny_d_dmcomm; localGridsizes[2] = pSPARC->Nx_d_dmcomm;
        int localGridStart[3]; // z, y, x
        localGridStart[0] = pSPARC->DMVertices_dmcomm[4]; localGridStart[1] = pSPARC->DMVertices_dmcomm[2]; localGridStart[2] = pSPARC->DMVertices_dmcomm[0];
        MPI_Datatype dataType = MPI_DOUBLE_COMPLEX;
        MPI_Datatype domainSubarray;
        MPI_Type_create_subarray(3, gridsizes, localGridsizes, localGridStart, MPI_ORDER_C, dataType, &domainSubarray);
        MPI_Type_commit(&domainSubarray);
        for (kpt = pSPARC->kpt_start_indx; kpt < pSPARC->kpt_end_indx + 1; kpt++) {
            if (kPqList[kpt][0] >= 0) continue; // this kpt is not in symmetric kpt list
            kptSym = -kPqList[kpt][0] - 1;
            for (band = pSPARC->band_start_indx; band < pSPARC->band_end_indx + 1; band++) {
                for (spin = pSPARC->spin_start_indx; spin < pSPARC->spin_end_indx + 1; spin++) {
                    read_orbitals_distributed_complex_RPA(pSPARC, 0, kpt, symk1[kptSym], symk2[kptSym], symk3[kptSym], band, spin, domainSubarray, pSPARC->Xorb_kpt);
                }
            }
        }
        MPI_Type_free(&domainSubarray);
    }
}


void read_orbitals_distributed_complex_RPA(SPARC_OBJ *pSPARC, int flagSymKpt, int kpt, double symk1, double symk2, double symk3, int band, int spin, MPI_Datatype domainSubarray, double _Complex *readXorb_kpt) {
    MPI_Status status;
    MPI_File orbitalFh;
    char orbitalFileName[100];
    int localSpin = spin - pSPARC->spin_start_indx;
    int localKpt = kpt - pSPARC->kpt_start_indx;
    int localBand = band - pSPARC->band_start_indx;
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int size_k = DMndsp * pSPARC->Nband_bandcomm;
    snprintf(orbitalFileName, 100, "orbitals/kpt_%.4f,%.4f,%.4f_band%d_spin%d.orbit", symk1, symk2, symk3, band, spin);
    printf("the name of the orbital file is %s\n", orbitalFileName);
    MPI_File_open(pSPARC->dmcomm, orbitalFileName, MPI_MODE_RDWR, MPI_INFO_NULL, &orbitalFh); // cannot create a file
    MPI_File_set_view(orbitalFh, 0, MPI_C_DOUBLE_COMPLEX, domainSubarray, "native", MPI_INFO_NULL);
    MPI_File_read_all(orbitalFh, readXorb_kpt + localKpt*size_k + localBand*DMndsp + localSpin*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, MPI_C_DOUBLE_COMPLEX, &status);
    if (!flagSymKpt) { // this k-point is not in symmetric kpt list, so the orbital it just read is the conjugate of this kpt
        for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
            *(readXorb_kpt + localKpt*size_k + localBand*DMndsp + localSpin*pSPARC->Nd_d_dmcomm + index) = conj(*(readXorb_kpt + localKpt*size_k + localBand*DMndsp + localSpin*pSPARC->Nd_d_dmcomm + index));
        }
    }
    MPI_File_close(&orbitalFh);
}

void restore_electronDensity(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld) {
    if (nuChi0EigscommIndex == -1) return;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    int flagPrintNuChi0Eigscomm = (nuChi0EigscommIndex == 0); // set which nuchi0Eigscomm will print output
#ifdef DEBUG
    if (flagPrintNuChi0Eigscomm && (rank == 0)) printf("restoring electron density ... \n");
#endif
    
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        int DMnd = pSPARC->Nd_d;
        if (flagPrintNuChi0Eigscomm) { // 0th nuChi0Eigscomm read electron density
            // for 1st Relax step/ MDstep, set initial electron density
            // read initial density from file 
            char inputDensFnames[3][L_STRING+L_PSD];
            // set up input density filename
            if (rank == 0) {
                char INPUT_DIR[L_PSD];
                extract_path_from_file(pSPARC->filename, INPUT_DIR, L_PSD);
                combine_path_filename(INPUT_DIR, pSPARC->InDensTCubFilename, inputDensFnames[0], L_STRING+L_PSD);
                combine_path_filename(INPUT_DIR, pSPARC->InDensUCubFilename, inputDensFnames[1], L_STRING+L_PSD);
                combine_path_filename(INPUT_DIR, pSPARC->InDensDCubFilename, inputDensFnames[2], L_STRING+L_PSD);
            }
            int nFileToRead = pSPARC->densfilecount;
            read_cube_and_dist_vec(
                pSPARC, inputDensFnames, pSPARC->electronDens, nFileToRead,
                pSPARC->DMVertices, pSPARC->dmcomm_phi
            );
            MPI_Bcast(pSPARC->electronDens, DMnd, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
        } else {  // other nuChi0Eigscomms receive electron density from 0th nuChi0Eigscomm
            MPI_Bcast(pSPARC->electronDens, DMnd, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
        }
        if (flagPrintNuChi0Eigscomm && (!rank)) {
            int densityIndex = 2;
            printf("I am rank %d in nuChi0Eigscomm %d; the %dth density is %9.6f\n", rank, nuChi0EigscommIndex, densityIndex, pSPARC->electronDens[densityIndex]);
        }
    }
}