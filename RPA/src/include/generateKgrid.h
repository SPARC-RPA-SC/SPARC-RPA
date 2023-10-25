#ifndef KPTGPT
#define KPTGPT

void transfer_kpoints(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

void recalculate_kpoints(SPARC_OBJ *pSPARC);

int set_qpoints(double *qptWts, double *q1, double *q2, double *q3, int Kx, int Ky, int Kz, double Lx, double Ly, double Lz);

void set_kPq_lists(int Nkpts_sym, double *k1sym, double *k2sym, double *k3sym, int Nkpts, double *k1, double *k2, double *k3, 
    int Nqpts_sym, double *q1, double *q2, double *q3, double Lx, double Ly, double Lz, int **kPqList);

int find_kpt_sym_index(double k1, double k2, double k3, int Nkpts_sym, double *k1sym, double *k2sym, double *k3sym, double Lx, double Ly, double Lz);

int find_kpt_sym_1to1(double k1Coord, double k2Coord, double k3Coord, int Nkpts_sym, double *k1sym, double *k2sym, double *k3sym);
#endif