#include "PS_FoxGlynn_OpenCL.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <iostream>
#include <iterator>

#include <vector>
#include <CL/cl.h>
#include "OpenCLUtil.h"

#include <math.h>
#include <prism.h>
#include <util.h>
#include <dd.h>
#include "sparse.h"
#include "jnipointer.h"
#include "PrismSparseGlob.h"

#define CL_PROF

template <typename T>
class PS_FoxGlynn_OpenCLKernel;

class FGWCoreCS : public OpenCLKernel
{
  public:
    FGWCoreCS(cl_device_id cl_device_, cl_context cl_context_, MatrixCS matrix, cl_real* fgw_ds)
      : OpenCLKernel(cl_device_, cl_context_, "", "FGWCoreCS") 
      , matrix_m(matrix)
    {
      cl_int err = 0;

      cl_uint n = matrix.n;
      cl_uint nzc = matrix.non_zero_cnt;
      // Create the buffers.
      cl_create_buffer<cl_real>(cl_msc_non_zero_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nzc, matrix.non_zero);
      cl_create_buffer<cl_uint>(cl_msc_non_zero_row_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nzc, matrix.non_zero_xs);
      cl_create_buffer<cl_uint>(cl_msc_col_offset_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n + 1, matrix.non_zero_ys_offset);

      cl_create_buffer<cl_real>(cl_fgw_d_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n, fgw_ds);

      cl_set_arg(0, cl_msc_non_zero_m);
      cl_set_arg(1, cl_msc_non_zero_row_m);
      cl_set_arg(2, cl_msc_col_offset_m);
      cl_set_arg(3, matrix_m.n);
      cl_set_arg(4, cl_fgw_d_m);
      // fgw_w
      cl_set_arg(6, cl_sum_m);
      cl_set_arg(7, cl_v0_m);
      cl_set_arg(8, cl_v1_m);
    }

    cl_kernel get_kernel() { return cl_kernel_m; }
    void set_fgw_w(cl_real w) { cl_set_arg(5, w); } 
    void set_veci_mem(cl_mem m) { cl_set_arg(7, m); }
    void set_veco_mem(cl_mem m) { cl_set_arg(8, m); }
    void set_sum_mem(cl_mem m) { cl_set_arg(6, m); }

  private:
    MatrixCS matrix_m;
    
    cl_mem cl_v0_m;
    cl_mem cl_v1_m;
    cl_mem cl_msc_non_zero_m;
    cl_mem cl_msc_non_zero_row_m;
    cl_mem cl_msc_col_offset_m;

    cl_mem cl_fgw_d_m;
    
    cl_mem cl_veci_m;
    cl_mem cl_veco_m;
    cl_mem cl_sum_m;

    size_t lws_m;
    size_t gws_m;
};

template <typename FGWCore>
class PS_FoxGlynn_OpenCLKernel 
{
  public:
    PS_FoxGlynn_OpenCLKernel
      ( cl_device_id cl_device_
      , cl_context cl_context_
      
      , MatrixCS matrix
      , cl_real* fgw_ds // fgw diagonals
      , cl_real* fgw_ws // fgw weights
      , cl_uint fgw_l 
      , cl_uint fgw_r

      , cl_real unif
      , bool is_cumul_reward
      )
      : cl_device_m(cl_device_)
      , cl_context_m(cl_context_)

      , core_m(cl_device_, cl_context_, matrix, fgw_ds)
      , matrix_m(matrix)
      , fgw_ws_m(fgw_ws)
      , fgw_l_m(fgw_l)
      , fgw_r_m(fgw_r)
      , fgw_i_m(1)

      , unif_m(unif)
      , is_cumul_reward_m(is_cumul_reward)
      {
        cl_int err = 0;

        cl_kernel_m = core_m.get_kernel();
        cl_queue_m = clCreateCommandQueue(cl_context_m, cl_device_m, 0, &err);
        cl_create_buffer<cl_real>(cl_context_m, cl_v0_m, CL_MEM_READ_WRITE, matrix_m.n);
        cl_create_buffer<cl_real>(cl_context_m, cl_v1_m, CL_MEM_READ_WRITE, matrix_m.n);
        cl_create_buffer<cl_real>(cl_context_m, cl_sum_m, CL_MEM_READ_WRITE, matrix_m.n);
        cl_fill_buffer<cl_real>(cl_queue_m, cl_sum_m, 0.0, matrix_m.n);
        core_m.set_sum_mem(cl_sum_m);
      }
   
    ~PS_FoxGlynn_OpenCLKernel()
    {
      clReleaseMemObject(cl_sum_m);
      clReleaseMemObject(cl_v1_m);
      clReleaseMemObject(cl_v0_m);
      clReleaseCommandQueue(cl_queue_m);
    }

    void sum(cl_real* x)
    {
      cl_read_buffer<cl_real>(cl_queue_m, cl_sum_m, matrix_m.n, x);
    }

    void run
      ( cl_real* vec_i
      , cl_real* vec_o
      , cl_uint times
      )
    {
      size_t lws_m = 256;
      size_t gws_m = least_greater_multiple(lws_m, (size_t)matrix_m.n);
      cl_write_buffer<cl_real>(cl_queue_m, cl_v0_m, matrix_m.n, vec_i);

      cl_mem& v0 = cl_v0_m;
      cl_mem& v1 = cl_v1_m;
      cl_event ev_iter_exec = NULL;
      for (cl_uint ii = 0; ii < times; ++ii)
      {
        if (ii != 0)
        {
          clWaitForEvents(1, &ev_iter_exec);
        }
        core_m.set_fgw_w(fgw_w());
        core_m.set_veci_mem(v0);
        core_m.set_veco_mem(v1);
        std::swap(v0, v1);

        clReleaseEvent(ev_iter_exec);
        clEnqueueNDRangeKernel(cl_queue_m, cl_kernel_m, 1, NULL, &gws_m, &lws_m, 0, NULL, &ev_iter_exec);

        ++fgw_i_m;
      }
      clWaitForEvents(1, &ev_iter_exec);
      cl_read_buffer<cl_real>(cl_queue_m, v0, matrix_m.n, vec_o);
      cl_read_buffer<cl_real>(cl_queue_m, v1, matrix_m.n, vec_i);
      clReleaseEvent(ev_iter_exec);
    }
    
  private:
    cl_real fgw_w()
    {
      if (fgw_i_m < fgw_l_m)
      {
        if (is_cumul_reward_m) { return 1.0 / unif_m; }
        else { return 0.0; }
      }
      else
      {
        return fgw_ws_m[fgw_i_m - fgw_l_m];
      }
    }

    cl_device_id cl_device_m;
    cl_context cl_context_m;
    cl_command_queue cl_queue_m;
    cl_kernel cl_kernel_m;

    FGWCore core_m;
    cl_mem cl_v0_m;
    cl_mem cl_v1_m;
    cl_mem cl_sum_m;
    
    MatrixCS matrix_m;
    cl_real* fgw_ws_m;
    cl_uint fgw_l_m;
    cl_uint fgw_r_m;
    cl_uint fgw_i_m;

    bool is_cumul_reward_m;
    cl_real unif_m;
};

void PS_FoxGlynn_OpenCL
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
  err = clGetDeviceIDs(cl_platform_id_m, CL_DEVICE_TYPE_GPU, 1, &cl_device_id_m, NULL); CLERR();
  cl_context_m = clCreateContext(0, 1, &cl_device_id_m, NULL, NULL, &err); CLERR();

  PS_FoxGlynn_OpenCLKernel<FGWCoreCS> kernel
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

