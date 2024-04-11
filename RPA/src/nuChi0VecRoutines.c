#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "main.h"
#include "restoreElectronicGroundState.h"
#include "sternheimerEquation.h"
#include "electrostatics_RPA.h"

void nuChi0_mult_vectors_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int omegaIndex, double *DVs, double *nuChi0DVs, int nuChi0EigsAmount, int flagNoDmcomm, int printFlag) {
#ifdef DEBUG    
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    double *sqrtNuDVs = pRPA->sprtNuDeltaVs;
    Calculate_sqrtNu_vecs_gamma(pSPARC, DVs, sqrtNuDVs, nuChi0EigsAmount, 0, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
#ifdef DEBUG    
    t2 = MPI_Wtime();
    if (!rank) printf("nuChi0Eigscomm %d, rank %d, omega %d, 1st Calculate_sqrtNu_vecs_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    FILE *outputFile;
    if (printFlag) {
        outputFile = fopen(pRPA->timeRecordname, "a");
        fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, 1st Calculate_sqrtNu_vecs_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
        fclose(outputFile);
    }
    t1 = MPI_Wtime();
#endif
    if (!flagNoDmcomm) {
        sternheimer_eq_gamma(pSPARC, pRPA, omegaIndex, nuChi0EigsAmount, sqrtNuDVs, printFlag);
    }
#ifdef DEBUG    
    t2 = MPI_Wtime();
    if (!rank) printf("nuChi0Eigscomm %d, rank %d, omega %d, sternheimer_eq_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    if (printFlag) {
        outputFile = fopen(pRPA->timeRecordname, "a");
        fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, sternheimer_eq_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
        fclose(outputFile);
    }
    t1 = MPI_Wtime();
#endif
    collect_deltaRho_gamma(pSPARC, pRPA->deltaRhos, nuChi0EigsAmount, 0, pRPA->nuChi0Eigscomm);
#ifdef DEBUG  
    t2 = MPI_Wtime();
    if (!rank) printf("nuChi0Eigscomm %d, rank %d, omega %d, collect_deltaRho_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    if (printFlag) {
        outputFile = fopen(pRPA->timeRecordname, "a");
        fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, collect_deltaRho_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
        fclose(outputFile);
    }
    t1 = MPI_Wtime();
#endif
    Calculate_sqrtNu_vecs_gamma(pSPARC, pRPA->deltaRhos, nuChi0DVs, nuChi0EigsAmount, 0, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
#ifdef DEBUG  
    t2 = MPI_Wtime();
    if (!rank) printf("nuChi0Eigscomm %d, rank %d, omega %d, 2nd Calculate_sqrtNu_vecs_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    if (printFlag) {
        outputFile = fopen(pRPA->timeRecordname, "a");
        fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, 2nd Calculate_sqrtNu_vecs_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
        fclose(outputFile);
    }
#endif
}

void nuChi0_mult_vectors_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int qptIndex, int omegaIndex, double _Complex *DVs_kpt, double _Complex *nuChi0DVs_kpt, int nuChi0EigsAmount, int flagNoDmcomm) {
    if (!flagNoDmcomm) {
        sternheimer_eq_kpt(pSPARC, pRPA, qptIndex, omegaIndex, nuChi0EigsAmount, DVs_kpt, 0);
    }
    collect_deltaRho_kpt(pSPARC, pRPA->deltaRhos_kpt, nuChi0EigsAmount, 0, pRPA->nuChi0Eigscomm);
    double qptx = pRPA->q1[qptIndex]; double qpty = pRPA->q2[qptIndex]; double qptz = pRPA->q3[qptIndex];
    Calculate_deltaRhoPotential_kpt(pSPARC, pRPA->deltaRhos_kpt, nuChi0DVs_kpt, qptx, qpty, qptz, nuChi0EigsAmount, 0, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
}