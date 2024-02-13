/**
 * @file    parallelization_RPA.c
 * @brief   This file contains parallelization function for RPA calculation
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <limits.h>
#define MKL_Complex16 double _Complex
#include "mkl.h"
#include "mkl_lapacke.h"
#include "blacs.h"     // Cblacs_*
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>

#include <initialization.h>

#include "main.h"
#include "parallelization_RPA.h"
#include "parallelization.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

void Setup_Comms_RPA(RPA_OBJ *pRPA, int Nspin, int Nkpts, int Nstates, int Nd) {
    int nprocWorld, rankWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    // The Sternheimer equation will be solved in pSPARC. pRPA is the structure saving variables not in pSPARC.
    dims_divide_Eigs(nprocWorld, rankWorld, pRPA->nuChi0Neig, Nspin, Nkpts, Nstates, Nd, &pRPA->npnuChi0Neig);
    // 1. q-point communicator, with its own q-point index (coords) and weight, saved in pRPA
    // 3. nuChi0Eigs communicator, distribute all trial eigenvectors of nuChi0, saved in pRPA
    pRPA->npnuChi0Neig = judge_npObject(pRPA->nuChi0Neig, nprocWorld, pRPA->npnuChi0Neig);
    pRPA->nNuChi0Eigscomm = distribute_comm_load(pRPA->nuChi0Neig, pRPA->npnuChi0Neig, rankWorld, nprocWorld, &(pRPA->nuChi0EigscommIndex), &(pRPA->nuChi0EigsStartIndex), &(pRPA->nuChi0EigsEndIndex));
    int color = (pRPA->nuChi0EigscommIndex >= 0) ? pRPA->nuChi0EigscommIndex : INT_MAX;
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &pRPA->nuChi0Eigscomm);

    pRPA->rank0nuChi0EigscommInWorld = rankWorld;
    MPI_Bcast(&pRPA->rank0nuChi0EigscommInWorld, 1, MPI_INT, 0, pRPA->nuChi0Eigscomm);
    #ifdef DEBUG
    printf("I am %d in comm world, I am in %d nuChi0Eigscomm, I will handle nuChi0Eigs %d ~ %d\n", rankWorld, pRPA->nuChi0EigscommIndex, pRPA->nuChi0EigsStartIndex, pRPA->nuChi0EigsEndIndex);
    #endif
    // 3.3 nuChi0Eigs bridge communicator, connect all processors in different nuChi0Eigs communicator having the same index, to broadcast eigenvalues and eigenvectors from DFT
    int nprocNuChi0EigsComm, rankNuChi0EigsComm;
    MPI_Comm_size(pRPA->nuChi0Eigscomm, &nprocNuChi0EigsComm);
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rankNuChi0EigsComm);
    int judgeJoinCompute = (pRPA->nuChi0EigscommIndex >= 0); // if it is 0, the processor will not join computation
    color = (judgeJoinCompute > 0) ? rankNuChi0EigsComm : INT_MAX;
    MPI_Comm_split(MPI_COMM_WORLD, color, pRPA->nuChi0EigscommIndex, &pRPA->nuChi0EigsBridgeComm);
    pRPA->nuChi0EigsBridgeCommIndex = (judgeJoinCompute > 0) ? rankNuChi0EigsComm : -1;
    int rankNuChi0EigsBridgeComm; 
    MPI_Comm_rank(pRPA->nuChi0EigsBridgeComm, &rankNuChi0EigsBridgeComm);
    #ifdef DEBUG
    printf("I am %d in comm world, I am in %d nuChi0EigsBridgeComm, My rank in it is %d\n", rankWorld, pRPA->nuChi0EigsBridgeCommIndex, rankNuChi0EigsBridgeComm);
    #endif
    
    // 4. every nuChi0Eigs communicator replace MPI_COMM_WORLD in Setup_Comms function of SPARC, call Setup_Comms_SPARC in RPA
    // 5. spin communicator, in pSPARC
    // 6. k-point communicator, use ALL k-points, no symmetric reduction, saved in SPARC
    // 7. band communicator, in pSPARC
    // 8. domain communicator, in pSPARC
}

void dims_divide_Eigs(int nprocWorld, int rankWorld, int nuChi0Neig, int Nspin, int Nkpts, int Nstates, int Nd, int *npnuChi0Neig) {
    // this function is for computing the optimal dividance on nuChi0Eigs, (spins, kpts and bands for pSPARC).
    int nproc_nuChi0Eigscomm = 0;
    if ((*npnuChi0Neig > nprocWorld) || (*npnuChi0Neig > nuChi0Neig)) { // confined by total number of processors!
        if (!rankWorld) printf("Input NP_NUCHI_EIGS_PARAL_RPA is larger than the total number of processors or number of eigenvalues nu chi0 operator to be solved,\n so it is not valid. This input is abandoned.\n");
        *npnuChi0Neig = -1;
    }
    if (*npnuChi0Neig <= 0) { // if there is valid input npnuChi0Neig, just apply it
        *npnuChi0Neig = (nprocWorld / (Nspin*Nkpts*Nstates)) < 1 ? 1 : (nprocWorld / (Nspin*Nkpts*Nstates)); // minumum value: 1
        if (*npnuChi0Neig > nuChi0Neig) *npnuChi0Neig = nuChi0Neig; // maximum value: nuChi0Neig
        if (Nd/(nuChi0Neig/(*npnuChi0Neig)) < 16) { // if the block size is too large, it may cause too few COCG max iteration time, damaging stability of future eigenproblem
            int goodBlockSize = Nd / 16;
            *npnuChi0Neig = nuChi0Neig / goodBlockSize + 1;
        }
    }
}

int judge_npObject(int nObjectInTotal, int sizeFatherComm, int npInput) {
    int npOutput = npInput;
    int maxLimit = min(sizeFatherComm, nObjectInTotal);
    if (npOutput == -1) {
        npOutput = maxLimit;
    } else if (npOutput > maxLimit) {
        npOutput = maxLimit;
    }
    return npOutput;
}

int distribute_comm_load(int nObjectInTotal, int npObject, int rankFatherComm, int sizeFatherComm, int *commIndex, int *objectStartIndex, int *objectEndIndex) {
    int sizeComm = sizeFatherComm / npObject;
    if (rankFatherComm < (sizeFatherComm - sizeFatherComm % npObject))
        *commIndex = rankFatherComm / sizeComm;
    else
        *commIndex = -1;
    
    int nEig, nObjectInComm; // for block cyclic
    nEig = (nObjectInTotal - 1) / npObject + 1; // this is equal to ceil(Nstates/npband), for int inputs only
    nObjectInComm = *commIndex < (nObjectInTotal / nEig) ? nEig : (*commIndex == (nObjectInTotal / nEig) ? (nObjectInTotal % nEig) : 0);

    if (*commIndex == -1) {
        *objectStartIndex = 0;
    } else if (*commIndex <= (nObjectInTotal / nEig)) {
        *objectStartIndex = *commIndex * nEig;
    } else {
        *objectStartIndex = nObjectInTotal;
    }
    *objectEndIndex = *objectStartIndex + nObjectInComm - 1;

    return nObjectInComm;
}

void setup_blacsComm_RPA(RPA_OBJ *pRPA, MPI_Comm SPARC_dmcomm_phi, int DMnd, int Nspinor_spincomm, int Nspinor_eig, 
    int Nd, int npNd, int MAX_NS, int eig_paral_blksz, int isGammaPoint) {
    int nprocWorld, rankWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int nprocNuChi0EigsComm = -1;
    int color;
    int dims[3] = {0, 0, 0};
    if (pRPA->nuChi0EigscommIndex < 0) {
        color = INT_MAX;
    }
    else if (SPARC_dmcomm_phi == MPI_COMM_NULL) {
        color = INT_MAX;
    } else {
        MPI_Comm_size(pRPA->nuChi0Eigscomm, &nprocNuChi0EigsComm);
        int rank_dmcomm_phi;
        MPI_Comm_rank(SPARC_dmcomm_phi, &rank_dmcomm_phi);
        color = rank_dmcomm_phi;
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, pRPA->nuChi0EigscommIndex, &pRPA->nuChi0BlacsComm);

    // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int size_blacscomm, DMndsp, DMndspe;
    int *usermap, *usermap_0, *usermap_1;
    int info, bandsizes[2], nprow, npcol, myrow, mycol;

    size_blacscomm = pRPA->npnuChi0Neig;
    DMndsp = DMnd * Nspinor_spincomm;
    DMndspe = DMnd * Nspinor_eig;

    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) {
        usermap = (int *)malloc(sizeof(int)*size_blacscomm);
        usermap_0 = (int *)malloc(sizeof(int)*size_blacscomm);
        usermap_1 = (int *)malloc(sizeof(int)*size_blacscomm);
        for (int i = 0; i < size_blacscomm; i++) {
            usermap[i] = usermap_0[i] = usermap_1[i] = rankWorld - pRPA->rank0nuChi0EigscommInWorld + i*nprocNuChi0EigsComm;
        }

        // in order to use a subgroup of blacscomm, use the following
        // to get a good number of subgroup processes
        bandsizes[0] = ((Nd-1)/npNd+1) * Nspinor_eig;
        bandsizes[1] = pRPA->nuChi0Neig;
        ScaLAPACK_Dims_2D_BLCYC(size_blacscomm, bandsizes, dims);
#ifdef DEBUG
        printf("global rank = %d, size_blacscomm = %d, nuChi0 ScaLAPACK topology %d, Dims = (%d, %d)\n", rankWorld, size_blacscomm, color, dims[0], dims[1]);
#endif
        // TODO: make it able to use a subgroup of the blacscomm! For now just enforce it.
        if (dims[0] * dims[1] != size_blacscomm) {
            dims[0] = size_blacscomm;
            dims[1] = 1;
        }
    } else {
        usermap = (int *)malloc(sizeof(int)*1);
        usermap_0 = (int *)malloc(sizeof(int)*1);
        usermap_1 = (int *)malloc(sizeof(int)*1);
        usermap[0] = usermap_0[0] = usermap_1[0] = rankWorld;
        dims[0] = dims[1] = 1;
    }

    int myrank_mpi, nprocs_mpi;
    Cblacs_pinfo( &myrank_mpi, &nprocs_mpi );

    // the following commands will create a context with handle ictxt_blacs
    Cblacs_get( -1, 0, &pRPA->ictxt_blacs );
    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) {
        Cblacs_gridmap( &pRPA->ictxt_blacs, usermap_0, 1, 1, pRPA->npnuChi0Neig); // row topology
    } else {
        Cblacs_gridmap( &pRPA->ictxt_blacs, usermap_0, 1, 1, dims[0] * dims[1]); // row topology
    }
    free(usermap_0);

    // create ictxt_blacs topology
    Cblacs_get( -1, 0, &pRPA->ictxt_blacs_topo );
    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) {
        // create usermap_1 = reshape(usermap,[dims[0], dims[1]])
        for (int j = 0; j < dims[1]; j++) {
            for (int i = 0; i < dims[0]; i++) {
                usermap_1[j*dims[0]+i] = usermap[i*dims[1]+j];
            }
        }
    }
    Cblacs_gridmap( &pRPA->ictxt_blacs_topo, usermap_1, dims[0], dims[0], dims[1] ); // Cart topology
    free(usermap_1);
    free(usermap);

    // get coord of each process in original context
    Cblacs_gridinfo( pRPA->ictxt_blacs, &nprow, &npcol, &myrow, &mycol );

    int ZERO = 0, mb, nb, llda;
    mb = max(1, DMndspe);
    nb = (pRPA->nuChi0Neig - 1) / pRPA->npnuChi0Neig + 1; // equal to ceil(Nstates/npband), for int only
    // set up descriptor for storage of orbitals in ictxt_blacs (original)
    llda = max(1, DMndsp);
    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) {
        descinit_(&pRPA->desc_orbitals[0], &DMndspe, &pRPA->nuChi0Neig,
                  &mb, &nb, &ZERO, &ZERO, &pRPA->ictxt_blacs, &llda, &info);
    } else {
        for (int i = 0; i < 9; i++)
            pRPA->desc_orbitals[i] = 0;
    }
#ifdef DEBUG
    int temp_r, temp_c;
    temp_r = numroc_( &DMndspe, &mb, &myrow, &ZERO, &nprow);
    temp_c = numroc_( &pRPA->nuChi0Neig, &nb, &mycol, &ZERO, &npcol);
    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) printf("global rank = %2d, 1D topo, my nuChi0 blacs rank = %d, BLCYC size (%d, %d), actual size (%d, %d), DMndspe %d, mb %d, myrow %d, nprow %d, nuChi0Neig %d, nb %d, mycol %d, npcol %d\n", 
        rankWorld, pRPA->nuChi0EigscommIndex, temp_r, temp_c, DMndsp, pRPA->nNuChi0Eigscomm,
        DMndspe, mb, myrow, nprow, pRPA->nuChi0Neig, nb, mycol, npcol);
#endif
    // get coord of each process in block cyclic topology context
    Cblacs_gridinfo( pRPA->ictxt_blacs_topo, &nprow, &npcol, &myrow, &mycol );
    pRPA->nprow_ictxt_blacs_topo = nprow;
    pRPA->npcol_ictxt_blacs_topo = npcol;

    // set up descriptor for block-cyclic format storage of orbitals in ictxt_blacs
    // TODO: make block-cyclic parameters mb and nb input variables!
    mb = max(1, DMndspe / dims[0]); // this is only block, no cyclic! Tune this to improve efficiency!
    nb = max(1, pRPA->nuChi0Neig / dims[1]); // this is only block, no cyclic!

    // find number of rows/cols of the local distributed orbitals
    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) {
        pRPA->nr_orb_BLCYC = numroc_( &DMndspe, &mb, &myrow, &ZERO, &nprow);
        pRPA->nc_orb_BLCYC = numroc_( &pRPA->nuChi0Neig, &nb, &mycol, &ZERO, &npcol);
    } else {
        pRPA->nr_orb_BLCYC = 1;
        pRPA->nc_orb_BLCYC = 1;
    }
    llda = max(1, pRPA->nr_orb_BLCYC);
    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) {
        descinit_(&pRPA->desc_orb_BLCYC[0], &DMndspe, &pRPA->nuChi0Neig,
                  &mb, &nb, &ZERO, &ZERO, &pRPA->ictxt_blacs_topo, &llda, &info);
    } else {
        for (int i = 0; i < 9; i++)
            pRPA->desc_orb_BLCYC[i] = 0;
    }
#ifdef DEBUG
    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) printf("global rank = %2d, 2D topo, my nuChi0 blacs rank = %d, BLCYC size (%d, %d), actual size (%d, %d), DMndspe %d, mb %d, myrow %d, nprow %d, nuChi0Neig %d, nb %d, mycol %d, npcol %d\n", 
        rankWorld, pRPA->nuChi0EigscommIndex, pRPA->nr_orb_BLCYC, pRPA->nc_orb_BLCYC, DMndsp, pRPA->nNuChi0Eigscomm,
        DMndspe, mb, myrow, nprow, pRPA->nuChi0Neig, nb, mycol, npcol);
#endif
    // set up distribution of projected Hamiltonian and the corresponding overlap matrix
    // TODO: Find optimal distribution of the projected Hamiltonian and mass matrix!
    //       For now Hp and Mp are distributed as follows: we distribute them in the same
    //       context topology as the bands.
    //       Note that mb = nb!

    // the maximum Nstates up to which we will use LAPACK to solve
    // the subspace eigenproblem in serial
    // int MAX_NS = 2000;
    pRPA->eig_useLAPACK = (pRPA->nuChi0Neig <= MAX_NS) ? 1 : 0; // just for test, 20

    int mbQ, nbQ, lldaQ;
       // block size for storing Hp and Mp
    if (pRPA->eig_useLAPACK == 1) {
        // in this case we will call LAPACK instead to solve the subspace eigenproblem
        mb = nb = pRPA->nuChi0Neig;
        mbQ = nbQ = 64; // block size for storing subspace eigenvectors
    } else {
        // in this case we will use ScaLAPACK to solve the subspace eigenproblem
        mb = nb = eig_paral_blksz;
        mbQ = nbQ = eig_paral_blksz; // block size for storing subspace eigenvectors
    }

    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) {
        pRPA->nr_Hp_BLCYC = pRPA->nr_Mp_BLCYC = numroc_( &pRPA->nuChi0Neig, &mb, &myrow, &ZERO, &nprow);
        pRPA->nr_Hp_BLCYC = pRPA->nr_Mp_BLCYC = max(1, pRPA->nr_Mp_BLCYC);
        pRPA->nc_Hp_BLCYC = pRPA->nc_Mp_BLCYC = numroc_( &pRPA->nuChi0Neig, &nb, &mycol, &ZERO, &npcol);
        pRPA->nc_Hp_BLCYC = pRPA->nc_Mp_BLCYC = max(1, pRPA->nc_Mp_BLCYC);
        pRPA->nr_Q_BLCYC = numroc_( &pRPA->nuChi0Neig, &mbQ, &myrow, &ZERO, &nprow);
        pRPA->nc_Q_BLCYC = numroc_( &pRPA->nuChi0Neig, &nbQ, &mycol, &ZERO, &npcol);
    } else {
        pRPA->nr_Hp_BLCYC = pRPA->nc_Hp_BLCYC = 1;
        pRPA->nr_Mp_BLCYC = pRPA->nc_Mp_BLCYC = 1;
        pRPA->nr_Q_BLCYC  = pRPA->nc_Q_BLCYC  = 1;
    }

    llda = max(1, pRPA->nr_Hp_BLCYC);
    lldaQ= max(1, pRPA->nr_Q_BLCYC);
    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) {
        descinit_(&pRPA->desc_Hp_BLCYC[0], &pRPA->nuChi0Neig, &pRPA->nuChi0Neig,
                  &mb, &nb, &ZERO, &ZERO, &pRPA->ictxt_blacs_topo, &llda, &info);
        for (int i = 0; i < 9; i++) {
            pRPA->desc_Mp_BLCYC[i] = pRPA->desc_Hp_BLCYC[i];
        }
        descinit_(&pRPA->desc_Q_BLCYC[0], &pRPA->nuChi0Neig, &pRPA->nuChi0Neig,
                  &mbQ, &nbQ, &ZERO, &ZERO, &pRPA->ictxt_blacs_topo, &lldaQ, &info);
    } else {
        for (int i = 0; i < 9; i++) {
            pRPA->desc_Q_BLCYC[i] = pRPA->desc_Mp_BLCYC[i] = pRPA->desc_Hp_BLCYC[i] = 0;
        }
    }

#ifdef DEBUG
    if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) printf("global rank = %d, Hp topo, mb = %d, mbQ = %d, myrow %d, nprow %d, mycol %d, npcol %d, nr_Hp = %d, nc_Hp = %d\n", rankWorld, mb, mbQ, myrow, nprow, mycol, npcol, pRPA->nr_Hp_BLCYC, pRPA->nc_Hp_BLCYC);
#endif

    // allocate memory for block cyclic distribution of projected Hamiltonian and mass matrix
    if (isGammaPoint){
        pRPA->Hp = (double *)malloc(pRPA->nr_Hp_BLCYC * pRPA->nc_Hp_BLCYC * sizeof(double));
        pRPA->Mp = (double *)malloc(pRPA->nr_Mp_BLCYC * pRPA->nc_Mp_BLCYC * sizeof(double));
        pRPA->Q  = (double *)malloc(pRPA->nr_Q_BLCYC * pRPA->nc_Q_BLCYC * sizeof(double));
    } else{
        pRPA->Hp_kpt = (double _Complex *) malloc(pRPA->nr_Hp_BLCYC * pRPA->nc_Hp_BLCYC * sizeof(double _Complex));
        pRPA->Mp_kpt = (double _Complex *) malloc(pRPA->nr_Mp_BLCYC * pRPA->nc_Mp_BLCYC * sizeof(double _Complex));
        pRPA->Q_kpt  = (double _Complex *) malloc(pRPA->nr_Q_BLCYC * pRPA->nc_Q_BLCYC * sizeof(double _Complex));
    }

    if (pRPA->eig_useLAPACK == 0) {
        if (pRPA->eig_paral_maxnp < 0) {
            char RorC, SorG;
            RorC = (isGammaPoint) ? 'R' : 'C';
            SorG = 'G';
            pRPA->eig_paral_maxnp = parallel_eigensolver_max_processor(pRPA->nuChi0Neig, RorC, SorG); // just for test
            // pRPA->eig_paral_maxnp = size_blacscomm;
        }
            
        int gridsizes[2] = {pRPA->nuChi0Neig, pRPA->nuChi0Neig}, ierr = 1;
        SPARC_Dims_create(min(size_blacscomm, pRPA->eig_paral_maxnp), 2, gridsizes, 1, pRPA->eig_paral_subdims, &ierr);
        if (ierr) pRPA->eig_paral_subdims[0] = pRPA->eig_paral_subdims[1] = 1;
#ifdef DEBUG
        if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) printf("global rank = %d, Maximun number of processors for RPA eigenvalue solver is %d\n", rankWorld, pRPA->eig_paral_maxnp);
        if ((pRPA->nuChi0EigscommIndex >= 0) && (SPARC_dmcomm_phi != MPI_COMM_NULL)) printf("global rank = %d, The dimension of subgrid for RPA eigen solver is (%d x %d).\n", rankWorld,
                                pRPA->eig_paral_subdims[0], pRPA->eig_paral_subdims[1]);
#endif
    }
    // #else
    // pRPA->eig_useLAPACK = 1;
    // #endif
}