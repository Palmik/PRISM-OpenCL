#include "PS_StochTransientKernel.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <iostream>

#include <vector>
#include <CL/cl.hpp>

#define CL_PROF

unsigned long int least_greater_multiple(unsigned long int a, unsigned long int min)
{
  unsigned long int r = a;
  while (r < min) { r += a; }
  return r;
}

char const* PS_StochTransientKernel::cl_program_source = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\r\n\r\ntypedef double real;\r\n\r\n__kernel void PS_StochTransientKernel\r\n  ( const uint warp_size\r\n  , __global real const* fw_non_zero\r\n  , __global uint const* fw_non_zero_row\r\n  , __global uint const* fw_seg_offset\r\n  , const uint fw_ns\r\n  , const uint fw_ns_rem\r\n\r\n  , __global real const* fgw_d\r\n  , __global real const* fgw_w\r\n  , __global real* sum\r\n\r\n  , __global real const* v0\r\n  , __global real* v1\r\n  )\r\n{\r\n  int col = get_group_id(0) * get_local_size(0) + get_local_id(0);\r\n  int seg_i = col / warp_size;\r\n  int off_i = get_local_id(0) % warp_size;\r\n\r\n  uint dim = (fw_ns - 1) * warp_size + fw_ns_rem;\r\n  if (col < dim)\r\n  {\r\n    real dot_product = fgw_d[col] * v0[col];\r\n    uint skip = (seg_i < fw_ns - 1) ? warp_size : fw_ns_rem;\r\n  \r\n    uint sb = fw_seg_offset[seg_i];\r\n    uint se = fw_seg_offset[seg_i + 1];\r\n    for (uint ii = sb + off_i; ii < se; ii += skip)\r\n    {\r\n      dot_product += fw_non_zero[ii] * v0[fw_non_zero_row[ii]];\r\n    }\r\n    v1[col] = dot_product;\r\n\r\n    sum[col] += fgw_w[0] * dot_product;\r\n  }\r\n}\r\n";

PS_StochTransientKernel::PS_StochTransientKernel
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

  , zero_m(0.0)
{
  std::cerr << dim_m << " " << msc_non_zero_size << std::endl;
  cl_int err = 0;

  // Create the basics.
  cl_queue_m = clCreateCommandQueue(cl_context_m, cl_device_m, 0, &err);
  cl_program_m = clCreateProgramWithSource(cl_context_m, 1, &PS_StochTransientKernel::cl_program_source, NULL, &err);
  err = clBuildProgram(cl_program_m, 1, &cl_device_m, NULL, NULL, NULL);
  cl_k0_m = clCreateKernel(cl_program_m, "PS_StochTransientKernel", &err);
  cl_k1_m = clCreateKernel(cl_program_m, "PS_StochTransientKernel", &err);

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
  cl_create_buffer<cl_real>(cl_fgw_w_m, CL_MEM_READ_ONLY, 1);
  cl_create_buffer<cl_real>(cl_sum_m, CL_MEM_READ_WRITE, dim_m);
  cl_fill_buffer<cl_real>(cl_sum_m, 0.0, dim_m);

  cl_uint fw_ns = (dim_m + (warp_size - 1)) / warp_size;
  cl_uint fw_ns_rem = (dim_m % warp_size) ? dim_m % warp_size : warp_size;

  cl_set_kernel_arg(cl_k0_m, 0, warp_size);
  cl_set_kernel_arg(cl_k0_m, 1, cl_fw_non_zero_m);
  cl_set_kernel_arg(cl_k0_m, 2, cl_fw_non_zero_row_m);
  cl_set_kernel_arg(cl_k0_m, 3, cl_fw_seg_offset_m);
  cl_set_kernel_arg(cl_k0_m, 4, fw_ns);
  cl_set_kernel_arg(cl_k0_m, 5, fw_ns_rem);
  cl_set_kernel_arg(cl_k0_m, 6, cl_fgw_d_m);
  cl_set_kernel_arg(cl_k0_m, 7, cl_fgw_w_m);
  cl_set_kernel_arg(cl_k0_m, 8, cl_sum_m);
  cl_set_kernel_arg(cl_k0_m, 9, cl_v0_m);
  cl_set_kernel_arg(cl_k0_m, 10, cl_v1_m);
  
  cl_set_kernel_arg(cl_k1_m, 0, warp_size);
  cl_set_kernel_arg(cl_k1_m, 1, cl_fw_non_zero_m);
  cl_set_kernel_arg(cl_k1_m, 2, cl_fw_non_zero_row_m);
  cl_set_kernel_arg(cl_k1_m, 3, cl_fw_seg_offset_m);
  cl_set_kernel_arg(cl_k1_m, 4, fw_ns);
  cl_set_kernel_arg(cl_k1_m, 5, fw_ns_rem);
  cl_set_kernel_arg(cl_k1_m, 6, cl_fgw_d_m);
  cl_set_kernel_arg(cl_k1_m, 7, cl_fgw_w_m);
  cl_set_kernel_arg(cl_k1_m, 8, cl_sum_m);
  cl_set_kernel_arg(cl_k1_m, 9, cl_v1_m);
  cl_set_kernel_arg(cl_k1_m, 10, cl_v0_m);
}

PS_StochTransientKernel::~PS_StochTransientKernel()
{
  clReleaseKernel(cl_k1_m);
  clReleaseKernel(cl_k0_m);
    
  clReleaseMemObject(cl_sum_m);
  clReleaseMemObject(cl_fgw_w_m);
  clReleaseMemObject(cl_fgw_d_m);
  clReleaseMemObject(cl_fw_seg_offset_m);
  clReleaseMemObject(cl_fw_non_zero_row_m);
  clReleaseMemObject(cl_fw_non_zero_m);
  clReleaseMemObject(cl_v1_m);
  clReleaseMemObject(cl_v0_m);
}

void PS_StochTransientKernel::run
  ( cl_real* vec_i
  , cl_real* vec_o
  , cl_uint times
  )
{
  cl_int err = 0;

  cl_real zero = 0.0;
  cl_write_buffer<cl_real>(cl_v0_m, dim_m, vec_i);

  cl_kernel& k0 = cl_k0_m;
  cl_kernel& k1 = cl_k1_m;

  cl_event ev_fgww = NULL;
  cl_event ev_iter_exec = NULL;
  for (cl_uint ii = 0; ii < times; ++ii)
  {
    if (ii == 0)
    {
      err = clEnqueueWriteBuffer(cl_queue_m, cl_fgw_w_m, CL_FALSE, 0, 1 * sizeof(cl_real), fgw_w_ptr(), 0, NULL, &ev_fgww); 
      //std::cerr << err << std::endl;
    }
    else
    {
      clReleaseEvent(ev_fgww);
      err = clEnqueueWriteBuffer(cl_queue_m, cl_fgw_w_m, CL_FALSE, 0, 1 * sizeof(cl_real), fgw_w_ptr(), 1, &ev_iter_exec, &ev_fgww); 
      //std::cerr << err << std::endl;
    }
    clReleaseEvent(ev_iter_exec);
    clEnqueueNDRangeKernel(cl_queue_m, k0, 1, NULL, &gws_m, &lws_m, 1, &ev_fgww, &ev_iter_exec);

    std::swap(k0, k1);

    ++fgw_i_m;
  }
  clWaitForEvents(1, &ev_iter_exec);
  cl_read_buffer<cl_real>((times % 2) ? cl_v1_m : cl_v0_m, dim_m, vec_o);
  clReleaseEvent(ev_fgww);
  clReleaseEvent(ev_iter_exec);
}

void PS_StochTransientKernel::sum(cl_real* x)
{
  cl_read_buffer<cl_real>(cl_sum_m, dim_m, x);
}

