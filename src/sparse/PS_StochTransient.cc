//==============================================================================
//  
//  Copyright (c) 2002-
//  Authors:
//  * Dave Parker <david.parker@comlab.ox.ac.uk> (University of Oxford, formerly University of Birmingham)
//  
//------------------------------------------------------------------------------
//  
//  This file is part of PRISM.
//  
//  PRISM is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//  
//  PRISM is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with PRISM; if not, write to the Free Software Foundation,
//  Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//  
//==============================================================================

// includes
#include "PrismSparse.h"
#include <math.h>
#include <prism.h>
#include "PS_StochTransientKernel.h"
#include "sparse.h"
#include "PrismSparseGlob.h"
#include "jnipointer.h"
#include <iostream>
#include <new>
#include <util.h>
#include <cudd.h>
#include <dd.h>
#include <odd.h>
#include <dv.h>
//------------------------------------------------------------------------------

JNIEXPORT jlong __jlongpointer JNICALL Java_sparse_PrismSparse_PS_1StochTransient
(
JNIEnv *env,
jclass cls,
jlong __jlongpointer tr,  // trans matrix
jlong __jlongpointer od,  // odd
jlong __jlongpointer in,  // initial distribution (note: this will be deleted afterwards)
jlong __jlongpointer rv,  // row vars
jint num_rvars,
jlong __jlongpointer cv,  // col vars
jint num_cvars,
jdouble time    // time bound
)
{
  // cast function parameters
  DdNode *trans = jlong_to_DdNode(tr);    // trans matrix
  ODDNode *odd = jlong_to_ODDNode(od);    // odd
  double *init = jlong_to_double(in);     // initial distribution
  DdNode **rvars = jlong_to_DdNode_array(rv); // row vars
  DdNode **cvars = jlong_to_DdNode_array(cv); // col vars

  // model stats
  int n;
  long nnz;
  // flags
  // sparse matrix
  CMSparseMatrix *cmsm = NULL;
  CMSCSparseMatrix *cmscsm = NULL;
  // vectors
  double *diags = NULL, *soln = NULL, *soln2 = NULL, *tmpsoln = NULL, *sum = NULL;
  DistVector *diags_dist = NULL;
  // fox glynn stuff
  FoxGlynnWeights fgw;
  // timing stuff
  long start1, start2, start3, stop;
  double time_taken, time_for_setup, time_for_iters;
  // misc
  bool done;
  int j, l, h;
  long i, iters, num_iters;
  double d, x, sup_norm, max_diag, weight, kb, kbt, unif, term_crit_param_unif;
  
  // exception handling around whole function
  try {
  
  // start clocks 
  start1 = start2 = util_cpu_time();
  
  // get number of states
  n = odd->eoff + odd->toff;
  
  // build sparse matrix
  PS_PrintToMainLog(env, "\nBuilding sparse matrix... ");

  cmsm = build_cm_sparse_matrix(ddman, trans, rvars, cvars, num_rvars, odd);
  nnz = cmsm->nnz;
  kb = cmsm->mem;
  kbt = kb;
  // print some info
  PS_PrintToMainLog(env, "[n=%d, nnz=%d] ", n, nnz);
  PS_PrintMemoryToMainLog(env, "[", kb, "]\n");
  
  // get vector of diagonals
  PS_PrintToMainLog(env, "Creating vector for diagonals... ");
  diags = cm_negative_row_sums(cmsm);
  kb = n*8.0/1024.0;
  kbt += kb;
  PS_PrintMemoryToMainLog(env, "[", kb, "]\n");
  
  // find max diagonal element
  max_diag = diags[0];
  for (i = 1; i < n; i++) if (diags[i] < max_diag) max_diag = diags[i];
  max_diag = -max_diag;
  
  // constant for uniformization
  unif = 1.02*max_diag;
  
  // modify diagonals
  for (i = 0; i < n; i++) diags[i] = diags[i] / unif + 1;
  
  // uniformization
  for (i = 0; i < nnz; i++) cmsm->non_zeros[i] /= unif;
  
  // create solution/iteration vectors
  PS_PrintToMainLog(env, "Allocating iteration vectors... ");
  // for soln, we just use init (since we are free to modify/delete this vector)
  // we also report the memory usage of this vector here, even though it has already been created
  soln = init;
  soln2 = new double[n];
  sum = new double[n];
  kb = n*8.0/1024.0;
  kbt += 3*kb;
  PS_PrintMemoryToMainLog(env, "[3 x ", kb, "]\n");
  
  // print total memory usage
  PS_PrintMemoryToMainLog(env, "TOTAL: [", kbt, "]\n");
  
  // compute new termination criterion parameter (epsilon/8)
  term_crit_param_unif = term_crit_param / 8.0;
  
  // compute poisson probabilities (fox/glynn)
  PS_PrintToMainLog(env, "\nUniformisation: q.t = %f x %f = %f\n", unif, time, unif * time);
  fgw = fox_glynn(unif * time, 1.0e-300, 1.0e+300, term_crit_param_unif);
  if (fgw.right < 0) throw "Overflow in Fox-Glynn computation (time bound too big?)";
  for (i = fgw.left; i <= fgw.right; i++) {
    fgw.weights[i-fgw.left] /= fgw.total_weight;
  }
  PS_PrintToMainLog(env, "Fox-Glynn: left = %ld, right = %ld\n", fgw.left, fgw.right);
  
  // set up vectors
  for (i = 0; i < n; i++) {
    sum[i] = 0.0;
  }
  
  // get setup time
  stop = util_cpu_time();
  time_for_setup = (double)(stop - start2)/1000;
  start2 = stop;
  start3 = stop;
  
  // start transient analysis
  done = false;
  num_iters = -1;
  PS_PrintToMainLog(env, "\nStarting iterations...\n");
  
  // if necessary, do 0th element of summation (doesn't require any matrix powers)
  if (fgw.left == 0) for (i = 0; i < n; i++) {
    sum[i] += fgw.weights[0] * soln[i];
  }
  
  // Set up OpenCL (platform, device, context)
  cl_int err = 0;

  cl_platform_id cl_platform_id_m;
  cl_device_id cl_device_id_m;
  cl_context cl_context_m;

  err = clGetPlatformIDs(1, &cl_platform_id_m, NULL);
  err = clGetDeviceIDs(cl_platform_id_m, CL_DEVICE_TYPE_GPU, 1, &cl_device_id_m, NULL);
  cl_context_m = clCreateContext(0, 1, &cl_device_id_m, NULL, NULL, &err);

  // Prepare the matrix and other data for the kernel.
  cl_uint* msc_column_offset = new cl_uint[cmsm->n + 1];
  if (cmsm->use_counts)
  {
    msc_column_offset[0] = 0;
    for (size_t ii = 1; ii <= cmsm->n; ++ii)
    {
      msc_column_offset[ii] = msc_column_offset[ii - 1] + cmsm->col_counts[ii - 1];
    }
  }
  else
  {
    for (size_t ii = 0; ii <= cmsm->n; ++ii)
    {
      msc_column_offset[ii] = cmsm->col_counts[ii];
    }
  }
  PS_StochTransientKernel kernel
    ( cl_device_id_m, cl_context_m

    , cmsm->non_zeros
    , (cl_uint*)cmsm->rows
    , msc_column_offset
    , (cl_uint)cmsm->nnz
    , (cl_uint)n
    
    , diags
    , fgw.weights
    , fgw.left
    );

  size_t iters_max_step = fgw.right / 2;
  for (iters = 0; (iters < fgw.right) && !done;)
  {
    size_t iters_step = (iters + iters_max_step < fgw.right) ? iters_max_step : fgw.right - iters;
    if (iters_step == 0) { break; }
    kernel.run(soln, soln2, iters_step);
    iters += iters_step;

    // check for steady state convergence
    if (do_ss_detect) {
      sup_norm = 0.0;
      for (i = 0; i < n; i++) {
        x = fabs(soln2[i] - soln[i]);
        if (term_crit == TERM_CRIT_RELATIVE) {
          x /= soln2[i];
        }
        if (x > sup_norm) sup_norm = x;
      }
      if (sup_norm < term_crit_param_unif) {
        done = true;
      }
    }
    
    // special case when finished early (steady-state detected)
    if (done) {
      kernel.sum(sum);
      // work out sum of remaining poisson probabilities
      if (iters <= fgw.left) {
        weight = 1.0;
      } else {
        weight = 0.0;
        for (i = iters; i <= fgw.right; i++) {
          weight += fgw.weights[i-fgw.left];
        }
      }
      // add to sum
      for (i = 0; i < n; i++) sum[i] += weight * soln2[i];
      PS_PrintToMainLog(env, "\nSteady state detected at iteration %ld\n", iters);
      num_iters = iters;
      break;
    }
    
    // print occasional status update
    if ((util_cpu_time() - start3) > UPDATE_DELAY) {
      PS_PrintToMainLog(env, "Iteration %d (of %d): ", iters, fgw.right);
      if (do_ss_detect) PS_PrintToMainLog(env, "max %sdiff=%f, ", (term_crit == TERM_CRIT_RELATIVE)?"relative ":"", sup_norm);
      PS_PrintToMainLog(env, "%.2f sec so far\n", ((double)(util_cpu_time() - start2)/1000));
      start3 = util_cpu_time();
    }
    
    // prepare for next iteration
    tmpsoln = soln;
    soln = soln2;
    soln2 = tmpsoln;
    
  }
  kernel.sum(sum);
  delete[] msc_column_offset;

  // stop clocks
  stop = util_cpu_time();
  time_for_iters = (double)(stop - start2)/1000;
  time_taken = (double)(stop - start1)/1000;
  
  // print iters/timing info
  if (num_iters == -1) num_iters = fgw.right;
  PS_PrintToMainLog(env, "\nIterative method: %ld iterations in %.2f seconds (average %.6f, setup %.2f)\n", num_iters, time_taken, time_for_iters/num_iters, time_for_setup);
  
  // catch exceptions: register error, free memory
  } catch (std::bad_alloc e) {
    PS_SetErrorMessage("Out of memory");
    if (sum) delete[] sum;
    sum = 0;
  } catch (const char *err) {
    PS_SetErrorMessage(err);
    if (sum) delete sum;
    sum = 0;
  } catch (cl::Error const& ex) {
    std::cout << "OpenCL exception: " << ex.what() << std::endl;
    PS_SetErrorMessage("OpenCL exception: ");
    PS_SetErrorMessage(ex.what());
  }

  
  // free memory
  if (cmscsm) delete cmscsm;
  if (cmsm) delete cmsm;
  if (diags) delete[] diags;
  if (diags_dist) delete diags_dist;
  // nb: we *do* free soln (which was originally init)
  if (soln) delete[] soln;
  if (soln2) delete[] soln2;
  
  return ptr_to_jlong(sum);
}

//------------------------------------------------------------------------------
