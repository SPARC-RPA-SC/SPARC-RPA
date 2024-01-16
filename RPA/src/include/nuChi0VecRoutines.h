#ifndef NUCHI0VECROUTINES
#define NUCHI0VECROUTINES

#include "isddft.h"
#include "main.h"

void nuChi0_mult_vectors_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int omegaIndex, double *DVs_phi, double *nuChi0DVs_phi, int nuChi0EigsAmount, int flagNoDmcomm);

void nuChi0_mult_vectors_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int qptIndex, int omegaIndex, double _Complex *DVs_kpt_phi, double _Complex *nuChi0DVs_kpt_phi, int nuChi0EigsAmount, int flagNoDmcomm);
#endif