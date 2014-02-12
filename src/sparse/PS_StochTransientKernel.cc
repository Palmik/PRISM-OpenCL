#include "PS_StochTransientKernel.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <iostream>

#include <vector>
#include <CL/cl.hpp>

unsigned long int least_greater_multiple(unsigned long int a, unsigned long int min)
{
  unsigned long int r = a;
  while (r < min) { r *= a; }
  return r;
}

char const* PS_StochTransientKernel::cl_kernel_source = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\r\n\r\n__kernel void PS_StochTransientKernel\r\n  ( const uint warp_size\r\n  , __global double const* fw_non_zero\r\n  , __global uint const* fw_non_zero_row\r\n  , __global uint const* fw_seg_offset\r\n  , const uint fw_ns\r\n  , const uint fw_ns_rem\r\n\r\n  , __global double const* fgw_d\r\n  , const double fgw_w\r\n  , __global double* sum\r\n\r\n  , __global double const* v0\r\n  , __global double* v1\r\n  )\r\n{\r\n  int col = get_group_id(0) * get_local_size(0) + get_local_id(0);\r\n  int seg_i = col / warp_size;\r\n  // We assume local_size > warp_size and that both are powers of 2. It should be equivalent to col % warp_size;\r\n  int off_i = get_local_id(0) & (warp_size - 1);\r\n\r\n  uint dim = (fw_ns - 1) * warp_size + fw_ns_rem;\r\n  if (col < dim)\r\n  {\r\n    double dot_product = fgw_d[col] * v0[col];\r\n    uint skip = (seg_i < fw_ns - 1) ? warp_size : fw_ns_rem;\r\n  \r\n    uint sb = fw_seg_offset[seg_i];\r\n    uint se = fw_seg_offset[seg_i + 1];\r\n    for (uint ii = sb + off_i; ii < se; ii += skip)\r\n    {\r\n      dot_product += fw_non_zero[ii] * v0[fw_non_zero_row[ii]];\r\n    }\r\n    v1[col] = dot_product;\r\n\r\n    sum[col] += fgw_w * dot_product;\r\n  }\r\n}\r\n";

PS_StochTransientKernel::PS_StochTransientKernel
  ( cl::Device& cl_device_
  , cl::Context& cl_context_

  , cl_double* msc_non_zero
  , cl_uint* msc_non_zero_row
  , cl_uint* msc_col_offset
  , cl_uint msc_non_zero_size
  , cl_uint dim

  , cl_double* fgw_d
  , cl_double* fgw_w_
  , cl_uint fgw_l
  )
  : cl_device_m(cl_device_)
  , cl_context_m(cl_context_)
  , cl_queue_m(cl_context(), cl_device(), CL_QUEUE_PROFILING_ENABLE, NULL)
  , cl_program_m(cl_context(), std::string(PS_StochTransientKernel::cl_kernel_source), true)
  , cl_kernel_m(cl_program(), "PS_StochTransientKernel")

  , dim_m(dim)
  , msc_non_zero_size_m(msc_non_zero_size)
  , fgw_w_m(fgw_w_)
  , fgw_l_m(fgw_l)
  , fgw_i_m(1)

  , cl_v0_m(cl_context(), CL_MEM_READ_WRITE, dim_m * sizeof(cl_double))
  , cl_v1_m(cl_context(), CL_MEM_READ_WRITE, dim_m * sizeof(cl_double))
  , cl_fgw_d_m(cl_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dim_m * sizeof(cl_double), fgw_d)
  , cl_sum_m(cl_context(), CL_MEM_READ_WRITE, dim_m * sizeof(cl_double))

  , lws_m(256)
  , gws_m(least_greater_multiple(lws_m, dim_m))
{
  cl_queue().enqueueFillBuffer(cl_sum_m, 0.0, 0, dim_m * sizeof(cl_double));
  cl_queue().finish();
  
  cl_uint warp_size = 64;

  std::vector<cl_double> fw_non_zero;
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

  cl_fw_non_zero_m = new cl::Buffer(cl_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fw_non_zero.size() * sizeof(cl_double), fw_non_zero.data());
  cl_fw_non_zero_row_m = new cl::Buffer(cl_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fw_non_zero_row.size() * sizeof(cl_double), fw_non_zero_row.data());
  cl_fw_seg_offset_m = new cl::Buffer(cl_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fw_seg_offset.size() * sizeof(cl_double), fw_seg_offset.data());

  cl_uint fw_ns = (dim_m + (warp_size - 1)) / warp_size;
  cl_uint fw_ns_rem = (dim_m % warp_size) ? dim_m % warp_size : warp_size;

  cl_kernel().setArg(0, warp_size);
  cl_kernel().setArg(1, *cl_fw_non_zero_m);
  cl_kernel().setArg(2, *cl_fw_non_zero_row_m);
  cl_kernel().setArg(3, *cl_fw_seg_offset_m);
  cl_kernel().setArg(4, fw_ns);
  cl_kernel().setArg(5, fw_ns_rem);
  cl_kernel().setArg(6, cl_fgw_d_m);
  // fgw_w
  cl_kernel().setArg(8, cl_sum_m);
  cl_kernel().setArg(9, cl_v0_m);
  cl_kernel().setArg(10, cl_v1_m);
}

PS_StochTransientKernel::~PS_StochTransientKernel()
{
  delete cl_fw_seg_offset_m;
  delete cl_fw_non_zero_row_m;
  delete cl_fw_non_zero_m;
}

void PS_StochTransientKernel::run
  ( cl_double* vec_i
  , cl_double* vec_o
  , cl_uint times
  )
{
  cl_queue().enqueueWriteBuffer(cl_v0_m, CL_TRUE, 0, dim_m * sizeof(cl_double), vec_i);

  cl::Buffer& v0 = cl_v0_m;
  cl::Buffer& v1 = cl_v1_m;

  cl_ulong prev_tend = 0;
  std::vector<cl::Event> ev_iter_exec(1);
  for (cl_uint ii = 0; ii < times; ++ii)
  {
    if (ii != 0)
    {
      cl::Event::waitForEvents(ev_iter_exec);

      cl_ulong tque, tsub, tbeg, tend;
      ev_iter_exec[0].getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &tque); 
      ev_iter_exec[0].getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &tsub); 
      ev_iter_exec[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &tbeg); 
      ev_iter_exec[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &tend);
      std::fprintf(stderr, "Iter %u:\nQUE: %lu\nSUB: %lu\nBEG: %lu\nEND: %lu\nQUE - PREV_TEND: %lu\nEND - QUE: %lu\n", fgw_i_m - 1, tque, tsub, tbeg, tend, tque - prev_tend, tend - tque);
      
      prev_tend = tend; 
    }
    cl_kernel().setArg(7, fgw_w());
    cl_kernel().setArg(9, v0);
    cl_kernel().setArg(10, v1);
    std::swap(v0, v1);

    cl_int err = cl_queue().enqueueNDRangeKernel(cl_kernel(), cl::NullRange, gws_m, lws_m, NULL, &ev_iter_exec[0]); 

    ++fgw_i_m;
  }
  cl_queue().enqueueReadBuffer(v0, CL_TRUE, 0, dim_m * sizeof(cl_double), vec_o, &ev_iter_exec);
}

void PS_StochTransientKernel::sum(cl_double* x)
{
  cl_queue().enqueueReadBuffer(cl_sum_m, CL_TRUE, 0, dim_m * sizeof(cl_double), x);
}

