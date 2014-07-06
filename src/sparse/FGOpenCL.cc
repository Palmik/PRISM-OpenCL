#include "FGOpenCL.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <iostream>
#include <iterator>

#include <math.h>
#include <prism.h>
#include <util.h>
#include <dd.h>
#include "sparse.h"
#include "jnipointer.h"
#include "PrismSparseGlob.h"
#include "PrismNativeGlob.h" // opencl_sparse_matrix_format command line option

#include <CL/cl.h>
#include "OpenCLUtil.h"

#include "FGOpenCL/Kernel.h"
#include "FGOpenCL/CoreCS.h"
#include "FGOpenCL/CoreCS_FW.h"

#define CL_PROF

void FGOpenCL
  ( JNIEnv* env
   
  , MatrixCS matrix
  , cl_real* fgw_ds // fgw diagonals
  , cl_real* fgw_ws // fgw weights
  , cl_uint fgw_l 
  , cl_uint fgw_r

  , cl_real* soln1 // in
  , cl_real* soln2 // out
  , cl_real* sum
 
  , long int time
  , cl_real unif
  
  , long int& num_iters

  , long int start2
  , long int start3
  
  , bool is_cumul_reward 
  )
{
  int fmt = opencl_sparse_matrix_format;
  std::cerr << fmt << std::endl;
  if (fmt == CL_SPARSE_MATRIX_FORMAT_CS_FW) {
    std::cerr << "FMT CS_FW" << std::endl;
    FGOpenCL_<FGCoreCS_FW>(env, matrix, fgw_ds, fgw_ws, fgw_l, fgw_r, soln1, soln2, sum, time, unif, num_iters, start2, start3, is_cumul_reward);
  }
  else { 
    std::cerr << "FMT CS" << std::endl;
    FGOpenCL_<FGCoreCS>(env, matrix, fgw_ds, fgw_ws, fgw_l, fgw_r, soln1, soln2, sum, time, unif, num_iters, start2, start3, is_cumul_reward);
  }
}

template <typename FGCore>
void FGOpenCL_
  ( JNIEnv* env
   
  , MatrixCS matrix
  , cl_real* fgw_ds // fgw diagonals
  , cl_real* fgw_ws // fgw weights
  , cl_uint fgw_l 
  , cl_uint fgw_r

  , cl_real* soln1 // in
  , cl_real* soln2 // out
  , cl_real* sum
 
  , long int time
  , cl_real unif
  
  , long int& num_iters

  , long int start2
  , long int start3
  
  , bool is_cumul_reward 
  )
{
  cl_int err = 0;

  cl_platform_id cl_platform_id_m;
  cl_device_id cl_device_id_m;
  cl_context cl_context_m;

  err = clGetPlatformIDs(1, &cl_platform_id_m, NULL); CLERR();
  
  err = clGetDeviceIDs(cl_platform_id_m, CL_DEVICE_TYPE_CPU, 1, &cl_device_id_m, NULL); CLERR();
  cl_context_m = clCreateContext(0, 1, &cl_device_id_m, NULL, NULL, &err); CLERR();
/*
  cl_device_information cpus[1]; size_t cpus_max_size = sizeof(cpus) / sizeof(cpus[0]); size_t cpus_size; 
  cl_device_information gpus[1]; size_t gpus_max_size = sizeof(gpus) / sizeof(gpus[0]); size_t gpus_size;
  err = cl_get_devices(cpus, cpus_max_size, cpus_size, CL_DEVICE_TYPE_CPU); CLERR();
  err = cl_get_devices(gpus, gpus_max_size, gpus_size, CL_DEVICE_TYPE_GPU); CLERR();
  cl_device_id_m = (gpus_size == 0) ? cpus[0].id : gpus[0].id; // TODO: Report error when both are 0;
*/

  FGKernel<FGCore> kernel
    ( cl_device_id_m
    , cl_context_m

    , matrix, fgw_ds, fgw_ws, fgw_l, fgw_r
    , unif
    , is_cumul_reward
    );

  double term_crit_param_unif = term_crit_param / 8.0;
  bool done = false;
  size_t iters_max_step = (do_ss_detect) ? 1 : fgw_r;
  for (size_t iters = 0; (iters < fgw_r) && !done;)
  {
    size_t iters_step = (iters + iters_max_step <= fgw_r) ? iters_max_step : fgw_r - iters;
    if (iters_step == 0) { break; }
    kernel.run(soln1, soln2, iters_step);
    iters += iters_step;

    // check for steady state convergence
    cl_real sup_norm;
    if (do_ss_detect)
    {
      sup_norm = 0.0;
      for (size_t i = 0; i < matrix.n; i++)
      {
        cl_real x = fabs(soln2[i] - soln1[i]);
        if (term_crit == TERM_CRIT_RELATIVE)
        {
          x /= soln2[i];
        }
        if (x > sup_norm) sup_norm = x;
      }
      if (sup_norm < term_crit_param_unif)
      {
        done = true;
      }
    }
    
    // special case when finished early (steady-state detected)
    if (done)
    {
      // work out sum of remaining poisson probabilities
      cl_real weight = (is_cumul_reward) ? time - iters / unif : 1.0;
      if (iters > fgw_l)
      {
        weight = 0.0;
        for (size_t i = iters; i <= fgw_r; i++)
        {
          weight += fgw_ws[i - fgw_l];
        }
      }
      // add to sum
      for (size_t i = 0; i < matrix.n; i++) sum[i] += weight * soln2[i];
      PS_PrintToMainLog(env, "\nSteady state detected at iteration %ld\n", iters);
      num_iters = iters + 1;
      break;
    }
    
    // print occasional status update
    if ((util_cpu_time() - start3) > UPDATE_DELAY)
    {
      PS_PrintToMainLog(env, "Iteration %d (of %d): ", iters, fgw_r);
      if (do_ss_detect) PS_PrintToMainLog(env, "max %sdiff=%f, ", (term_crit == TERM_CRIT_RELATIVE)?"relative ":"", sup_norm);
      PS_PrintToMainLog(env, "%.2f sec so far\n", ((double)(util_cpu_time() - start2)/1000));
      start3 = util_cpu_time();
    }
    
    // prepare for next iteration
    cl_real* tmpsoln = soln1;
    soln1 = soln2;
    soln2 = tmpsoln;
    kernel.sum(sum);
  }
}

