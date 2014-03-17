#include "PS_FoxGlynn_OpenCL.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <iostream>

#include <vector>
#include <CL/cl.hpp>

#include <math.h>
#include <prism.h>
#include <util.h>
#include <dd.h>
#include "sparse.h"
#include "jnipointer.h"
#include "PrismSparseGlob.h"

#define CL_PROF

void PS_FoxGlynn_OpenCL
  ( JNIEnv* env
  
  , cl_real* msc_non_zero
  , cl_uint* msc_non_zero_row
  , cl_uint* msc_col_offset
  , cl_uint msc_non_zero_size
  , cl_uint msc_dim

  , cl_real* fgw_ds
  , cl_real* fgw_ws
  , cl_uint fgw_l
  , cl_uint fgw_r

  , cl_real* soln1
  , cl_real* soln2
  , cl_real* sum

  , long int& num_iters

  , long int start2
  , long int start3 
  )
{
  double term_crit_param_unif = term_crit_param / 8.0;
  
  // Set up OpenCL (platform, device, context)
  cl_int err = 0;

  cl_platform_id cl_platform_id_m;
  cl_device_id cl_device_id_m;
  cl_context cl_context_m;

  err = clGetPlatformIDs(1, &cl_platform_id_m, NULL);
  err = clGetDeviceIDs(cl_platform_id_m, CL_DEVICE_TYPE_GPU, 1, &cl_device_id_m, NULL);
  cl_context_m = clCreateContext(0, 1, &cl_device_id_m, NULL, NULL, &err);

  PS_FoxGlynn_OpenCLKernel kernel
    ( cl_device_id_m, cl_context_m

    , msc_non_zero 
    , msc_non_zero_row
    , msc_col_offset
    , msc_non_zero_size
    , msc_dim
    
    , fgw_ds
    , fgw_ws
    , fgw_l
    );

  bool done = false;
  size_t iters_max_step = fgw_r / 2;
  for (size_t iters = 0; (iters < fgw_r) && !done;)
  {
    size_t iters_step = (iters + iters_max_step < fgw_r) ? iters_max_step : fgw_r - iters;
    if (iters_step == 0) { break; }
    kernel.run(soln1, soln2, iters_step);
    iters += iters_step;

    // check for steady state convergence
    cl_real sup_norm;
    if (do_ss_detect)
    {
      sup_norm = 0.0;
      for (size_t i = 0; i < msc_dim; i++)
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
      kernel.sum(sum);
      // work out sum of remaining poisson probabilities
      cl_real weight = 1.0;
      if (iters > fgw_l)
      {
        weight = 0.0;
        for (size_t i = iters; i <= fgw_r; i++)
        {
          weight += fgw_ws[i - fgw_l];
        }
      }
      // add to sum
      for (size_t i = 0; i < msc_dim; i++) sum[i] += weight * soln2[i];
      PS_PrintToMainLog(env, "\nSteady state detected at iteration %ld\n", iters);
      num_iters = iters;
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
    
  }
  kernel.sum(sum);
}
unsigned long int least_greater_multiple(unsigned long int a, unsigned long int min)
{
  unsigned long int r = a;
  while (r < min) { r += a; }
  return r;
}

char const* PS_FoxGlynn_OpenCLKernel::cl_program_source = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\r\n\r\ntypedef double real;\r\n\r\n__kernel void PS_FoxGlynn\r\n  ( const uint warp_size\r\n  , __global real const* fw_non_zero\r\n  , __global uint const* fw_non_zero_row\r\n  , __global uint const* fw_seg_offset\r\n  , const uint fw_ns\r\n  , const uint fw_ns_rem\r\n\r\n  , __global real const* fgw_d\r\n  , const real fgw_w\r\n  , __global real* sum\r\n\r\n  , __global real const* v0\r\n  , __global real* v1\r\n  )\r\n{\r\n  int col = get_group_id(0) * get_local_size(0) + get_local_id(0);\r\n  int seg_i = col / warp_size;\r\n  int off_i = get_local_id(0) % warp_size;\r\n\r\n  uint dim = (fw_ns - 1) * warp_size + fw_ns_rem;\r\n  if (col < dim)\r\n  {\r\n    real dot_product = fgw_d[col] * v0[col];\r\n    uint skip = (seg_i < fw_ns - 1) ? warp_size : fw_ns_rem;\r\n  \r\n    uint sb = fw_seg_offset[seg_i];\r\n    uint se = fw_seg_offset[seg_i + 1];\r\n    for (uint ii = sb + off_i; ii < se; ii += skip)\r\n    {\r\n      dot_product = fma(fw_non_zero[ii], v0[fw_non_zero_row[ii]], dot_product);\r\n    }\r\n    v1[col] = dot_product;\r\n\r\n    sum[col] = fma(fgw_w, dot_product, sum[col]);\r\n  }\r\n}\r\n";

PS_FoxGlynn_OpenCLKernel::PS_FoxGlynn_OpenCLKernel
  ( cl_device_id cl_device_
  , cl_context cl_context_

  , cl_real* msc_non_zero
  , cl_uint* msc_non_zero_row
  , cl_uint* msc_col_offset
  , cl_uint msc_non_zero_size
  , cl_uint dim

  , cl_real* fgw_d_
  , cl_real* fgw_w_
  , cl_uint fgw_l_
  )
  : cl_device_m(cl_device_)
  , cl_context_m(cl_context_)

  , dim_m(dim)
  , msc_non_zero_size_m(msc_non_zero_size)
  , fgw_w_m(fgw_w_)
  , fgw_l_m(fgw_l_)
  , fgw_i_m(1)

  , lws_m(256)
  , gws_m(least_greater_multiple(lws_m, dim_m))
{
  cl_int err = 0;

  // Create the basics.
  cl_queue_m = clCreateCommandQueue(cl_context_m, cl_device_m, 0, &err);
  cl_program_m = clCreateProgramWithSource(cl_context_m, 1, &PS_FoxGlynn_OpenCLKernel::cl_program_source, NULL, &err);
  err = clBuildProgram(cl_program_m, 1, &cl_device_m, NULL, NULL, NULL);
  cl_kernel_m = clCreateKernel(cl_program_m, "PS_FoxGlynn", &err);

  // Compute the MSC full-warp representation.
  cl_uint warp_size = 64;

  std::vector<cl_real> fw_non_zero;
  fw_non_zero.reserve(msc_non_zero_size);
  std::vector<cl_uint> fw_non_zero_row;
  fw_non_zero_row.reserve(msc_non_zero_size);
  std::vector<cl_uint> fw_seg_offset;
  fw_seg_offset.reserve((dim_m + (warp_size - 1)) / warp_size);
  fw_seg_offset.push_back(0);
  for (cl_uint ii = 0; ii < dim_m; )
  {
    // Step size -- how many columns are in this segment. Always LWS, except maybe for the last segment.
    cl_uint ss = std::min(warp_size, dim_m - ii);
   
    cl_uint mr = 0;
    for (size_t ic = ii; ic < ii + ss; ++ic)
    {
      mr = std::max(mr, msc_col_offset[ic + 1] - msc_col_offset[ic]);
    }

    for (size_t ir = 0; ir < mr; ++ir)
    {
      for (size_t ic = ii; ic < ii + ss; ++ic)
      {
        if (ir < msc_col_offset[ic + 1] - msc_col_offset[ic])
        {
          size_t i = msc_col_offset[ic] + ir;
          fw_non_zero.push_back(msc_non_zero[i]);
          fw_non_zero_row.push_back(msc_non_zero_row[i]);
        }
        else
        {
          fw_non_zero.push_back(0.0);
          fw_non_zero_row.push_back(0);
        }
      }
    }   
    fw_seg_offset.push_back(fw_non_zero.size()); 
    ii += ss;
  }

  // Create the buffers.
  cl_create_buffer<cl_real>(cl_v0_m, CL_MEM_READ_WRITE, dim_m);
  cl_create_buffer<cl_real>(cl_v1_m, CL_MEM_READ_WRITE, dim_m);

  cl_create_buffer<cl_real>(cl_fw_non_zero_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fw_non_zero.size(), fw_non_zero.data());
  cl_create_buffer<cl_uint>(cl_fw_non_zero_row_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fw_non_zero_row.size(), fw_non_zero_row.data());
  cl_create_buffer<cl_uint>(cl_fw_seg_offset_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fw_seg_offset.size(), fw_seg_offset.data());

  cl_create_buffer<cl_real>(cl_fgw_d_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dim_m, fgw_d_);
  cl_create_buffer<cl_real>(cl_sum_m, CL_MEM_READ_WRITE, dim_m);
  cl_fill_buffer<cl_real>(cl_sum_m, 0.0, dim_m);

  cl_uint fw_ns = (dim_m + (warp_size - 1)) / warp_size;
  cl_uint fw_ns_rem = (dim_m % warp_size) ? dim_m % warp_size : warp_size;

  cl_set_kernel_arg(cl_kernel_m, 0, warp_size);
  cl_set_kernel_arg(cl_kernel_m, 1, cl_fw_non_zero_m);
  cl_set_kernel_arg(cl_kernel_m, 2, cl_fw_non_zero_row_m);
  cl_set_kernel_arg(cl_kernel_m, 3, cl_fw_seg_offset_m);
  cl_set_kernel_arg(cl_kernel_m, 4, fw_ns);
  cl_set_kernel_arg(cl_kernel_m, 5, fw_ns_rem);
  cl_set_kernel_arg(cl_kernel_m, 6, cl_fgw_d_m);
  // fgw_w
  cl_set_kernel_arg(cl_kernel_m, 8, cl_sum_m);
  cl_set_kernel_arg(cl_kernel_m, 9, cl_v0_m);
  cl_set_kernel_arg(cl_kernel_m, 10, cl_v1_m);
}

PS_FoxGlynn_OpenCLKernel::~PS_FoxGlynn_OpenCLKernel()
{
  clReleaseKernel(cl_kernel_m);
    
  clReleaseMemObject(cl_sum_m);
  clReleaseMemObject(cl_fgw_d_m);
  clReleaseMemObject(cl_fw_seg_offset_m);
  clReleaseMemObject(cl_fw_non_zero_row_m);
  clReleaseMemObject(cl_fw_non_zero_m);
  clReleaseMemObject(cl_v1_m);
  clReleaseMemObject(cl_v0_m);
}

void PS_FoxGlynn_OpenCLKernel::run
  ( cl_real* vec_i
  , cl_real* vec_o
  , cl_uint times
  )
{
  cl_write_buffer<cl_real>(cl_v0_m, dim_m, vec_i);

  cl_mem& v0 = cl_v0_m;
  cl_mem& v1 = cl_v1_m;

  cl_event ev_iter_exec = NULL;
  for (cl_uint ii = 0; ii < times; ++ii)
  {
    if (ii != 0)
    {
      clWaitForEvents(1, &ev_iter_exec);
    }
    cl_set_kernel_arg(cl_kernel_m, 7, fgw_w());
    cl_set_kernel_arg(cl_kernel_m, 9, v0);
    cl_set_kernel_arg(cl_kernel_m, 10, v1);
    std::swap(v0, v1);

    clReleaseEvent(ev_iter_exec);
    clEnqueueNDRangeKernel(cl_queue_m, cl_kernel_m, 1, NULL, &gws_m, &lws_m, 0, NULL, &ev_iter_exec);

    ++fgw_i_m;
  }
  clWaitForEvents(1, &ev_iter_exec);
  cl_read_buffer<cl_real>(v0, dim_m, vec_o);
  clReleaseEvent(ev_iter_exec);
}

void PS_FoxGlynn_OpenCLKernel::sum(cl_real* x)
{
  cl_read_buffer<cl_real>(cl_sum_m, dim_m, x);
}

