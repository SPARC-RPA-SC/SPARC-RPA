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

void restore_electronicGroundState(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld) {
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
        restore_orbitals(pSPARC, nuChi0Eigscomm, nuChi0EigsBridgeComm, nuChi0EigscommIndex, rank0nuChi0EigscommInWorld);
    }

    // initialize electron density rho (initial guess)
    restore_electronDensity(pSPARC, nuChi0Eigscomm, nuChi0EigsBridgeComm, nuChi0EigscommIndex, rank0nuChi0EigscommInWorld);
    
}

void restore_orbitals(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld) {
    if ((nuChi0EigscommIndex == -1) || (pSPARC->dmcomm == MPI_COMM_NULL)) return; // not take part in computation
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    int flagPrintNuChi0Eigscomm = (nuChi0EigscommIndex == 0); // set which nuchi0Eigscomm will print output
#ifdef DEBUG
    if (rank == 0) printf("restoring Kohn-Sham orbitals ... \n");
#endif

    int k, n, DMnd, DMndsp, size_k, len_tot, spinor;
#ifdef DEBUG
    double t1, t2;
#endif
    // Multiply a factor for a spinor wavefunction
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    size_k = DMndsp * pSPARC->Nband_bandcomm;
    // notice that in processors not for orbital calculations len_tot = 0
    len_tot = size_k * pSPARC->Nkpts_kptcomm;
    
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};

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
    }        
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Finished setting random orbitals. Time taken: %.3f ms\n",(t2-t1)*1e3);
#endif

    if (nuChi0EigscommIndex == 0) { // read orbitals, then broadcast through nuChi0EigsBridgeCommIndex

    } else { // receiver of broadcast

    }
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
        if (!rank) {
            int densityIndex = 2;
            printf("I am rank %d in nuChi0Eigscomm %d; the %dth density is %9.6f\n", rank, nuChi0EigscommIndex, densityIndex, pSPARC->electronDens[densityIndex]);
        }
    }
}