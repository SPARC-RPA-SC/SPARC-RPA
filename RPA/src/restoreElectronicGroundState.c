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

#include "readScf.h"

void restore_electronicGroundState(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld) {
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
        
        // initialize orbitals psi
        Init_orbital(pSPARC);
    }

    // initialize electron density rho (initial guess)
    Init_electronDensity(pSPARC);

    read_scf(pSPARC, nuChi0EigscommIndex);
    
}