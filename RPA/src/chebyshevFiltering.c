#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "hamiltonianVecRoutines.h"

#include "main.h"
#include "chebyshevFiltering.h"

void chebyshev_filtering(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1) return;
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    // Sternheimer equation should be solved in RPA_OBJ, in other words, it should not use pSPARC.
    // To generate Hamiltonian in RPA domain topologies, it is necessary to generate Laplacian*Vec independent of pSPARC.
    // Then it can be called easily by any other feature. The transformation is done in function prepare_Hamiltonian(&SPARC, &RPA)
    if (pSPARC->dmcomm != MPI_COMM_NULL) {
        double *testHxResults = (double*)calloc(sizeof(double), pSPARC->Nkpts_kptcomm*pSPARC->Nspin_spincomm*pSPARC->Nband_bandcomm);
        test_Hx(pSPARC, testHxResults);
        if (nuChi0EigscommIndex == 0) {
            printf("rank %d in nuChi0Eigscomm %d, the relative error epsilon*psi - H*psi of the first two bands are %.6E %.6E\n", rank, nuChi0EigscommIndex,
                 testHxResults[0], testHxResults[1]);
        }
        free(testHxResults);
    }
}

void test_Hx(SPARC_OBJ *pSPARC, double *testHxResults) { // make a test to see the accuracy of eigenvalues and eigenvectors saved in every nuChi0Eigscomm
    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;    
    int *DMVertices = pSPARC->DMVertices_dmcomm;
    int ncol = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    MPI_Comm comm = pSPARC->dmcomm;
    if (pSPARC->isGammaPoint) { // follow the sequence in function Calculate_elecDens, divide gamma-point, k-point first
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) { // follow the sequence in function eigSolve_CheFSI, divide spin
            int sg  = pSPARC->spin_start_indx + spn_i;
            Hamiltonian_vectors_mult(
                pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
                pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, 0, pSPARC->Xorb + spn_i*DMnd, DMndsp, pSPARC->Yorb + spn_i*DMnd, DMndsp, spn_i, comm
            );
            for (int bandIndex = 0; bandIndex < ncol; bandIndex++) { // verify the correctness of psi
                double *psi = pSPARC->Xorb + bandIndex*DMndsp + spn_i*DMnd;
                double *Hpsi = pSPARC->Yorb + bandIndex*DMndsp + spn_i*DMnd;
                double eigValue = pSPARC->lambda[spn_i*ncol + bandIndex];
                for (int i = 0; i < DMnd; i++) {
                    testHxResults[spn_i*ncol + bandIndex] += (*(psi + i) * eigValue - *(Hpsi + i)) * (*(psi + i) * eigValue - *(Hpsi + i));
                }
            }
        }
    }
    else {
        int size_k = DMndsp * pSPARC->Nband_bandcomm;
        for(int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) { // follow the sequence in function eigSolve_CheFSI_kpt
            int sg  = pSPARC->spin_start_indx + spn_i;
            for (int kpt = 0; kpt < Nkpts_kptcomm; kpt++) {
                Hamiltonian_vectors_mult_kpt(
                    pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
                    pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, 0, pSPARC->Xorb_kpt + kpt*size_k + spn_i*DMnd, DMndsp, pSPARC->Yorb_kpt + spn_i*DMnd, DMndsp, spn_i, kpt, comm
                );
                for (int bandIndex = 0; bandIndex < ncol; bandIndex++) { // verify the correctness of psi
                    double _Complex *psi = pSPARC->Xorb_kpt + kpt*size_k + bandIndex*DMndsp + spn_i*DMnd;
                    double _Complex *Hpsi = pSPARC->Yorb_kpt + bandIndex*DMndsp + spn_i*DMnd;
                    double eigValue = pSPARC->lambda[spn_i*Nkpts_kptcomm*ncol + kpt*ncol + bandIndex];
                    for (int i = 0; i < DMnd; i++) {
                        testHxResults[spn_i*Nkpts_kptcomm*ncol + kpt*ncol + bandIndex] += (*(psi + i) * eigValue - *(Hpsi + i)) * (*(psi + i) * eigValue - *(Hpsi + i));
                    }
                }
            }
        }
    }
}