#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "main.h"
#include "printResult.h"

void print_result(RPA_OBJ *pRPA, int nAtoms) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
        double Erpa = 0.0;
        FILE *output_fp = fopen(pRPA->filename_out,"a");
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"Energy terms in every (qpt, omega) pair (Ha)\n");
        for (int qptIndex = 0; qptIndex < pRPA->Nqpts_sym; qptIndex++) {
            fprintf(output_fp,"q-point %d\n", qptIndex + 1);
            for (int omegaIndex = 0; omegaIndex < pRPA->Nomega; omegaIndex++) {
                double ErpaTerm = pRPA->ErpaTerms[qptIndex*pRPA->Nomega + omegaIndex];
                fprintf(output_fp,"omega %d: %.5E, ", omegaIndex + 1, ErpaTerm);
                Erpa += ErpaTerm;
                if (omegaIndex % 3 == 2) fprintf(output_fp,"\n");
            }
            fprintf(output_fp,"\n");
        }
        fprintf(output_fp,"Total RPA correlation energy: %.5E (Ha), %.5E (Ha/atom)\n", Erpa, Erpa / nAtoms);
        fclose(output_fp);
    }
}