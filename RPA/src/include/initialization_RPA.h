#ifndef INITIALRPA 
#define INITIALRPA

#include "isddft.h"

void initialize_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int argc, char* argv[]);

void transfer_kpoints(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void recalculate_kpoints(SPARC_OBJ *pSPARC);

void RPA_Input_MPI_create(MPI_Datatype *RPA_INPUT_MPI);

void set_RPA_defaults(RPA_INPUT_OBJ *pRPA_Input, int Nstates, int Nd);

void read_RPA_inputs(RPA_INPUT_OBJ *pRPA_Input);

void RPA_copy_inputs(RPA_OBJ *pRPA, RPA_INPUT_OBJ *pRPA_Input);

int set_qpoints(double *qptWts, double *q1, double *q2, double *q3, int Kx, int Ky, int Kz, double Lx, double Ly, double Lz);

void set_kPq_lists(int Nkpts_sym, double *k1sym, double *k2sym, double *k3sym, int Nkpts, double *k1, double *k2, double *k3, 
    int nqpts_sym, double *q1, double *q2, double *q3, int **kPqList);

void set_omegas(double *omega, double *omega01, double *omegaWts, int Nomega);

void write_settings(RPA_OBJ *pRPA);

#endif