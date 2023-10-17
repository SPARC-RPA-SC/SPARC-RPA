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

#include "main.h"
#include "parallelization_RPA.h"
#include "parallelization.h"

void Setup_Comms_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    // The Sternheimer equation will be solved in pSPARC. pRPA is the structure saving variables not in pSPARC.
    // After wavefunctions \psi in reduced kpt grid are read by using the initialized pSPARC,
    // the communicators in pSPARC will be totally rebuilt.
    // every spin communicator in pSPARC->spincomm in RPA is for dividing pairs of (spin, omega), pSPARC->Nspin = spin number * omega number
    // every kpt communicator in pSPARC->kptcomm in RPA is for dividing pairs of (k-point, q-point), pSPARC->Nkpts = all kpt number * Nkpt_sym
    // then go to band and domain dividence.
}