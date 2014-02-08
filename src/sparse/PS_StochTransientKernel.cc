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
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\r\n\r\n__kernel void PS_StochTransientKernel\r\n  ( __global double const* msc_non_zero\r\n  , __global uint const* msc_non_zero_row\r\n  , __global uint const* msc_column_offset\r\n  , const uint msc_dim\r\n\r\n  , __global double const* fgw_d\r\n  , const double fgw_w\r\n  , __global double* sum\r\n\r\n  , __global double const* v0\r\n  , __global double* v1\r\n  )\r\n{\r\n\tint col = get_global_id(0);\r\n\tif (col < msc_dim)\r\n\t{\r\n    uint cb = msc_column_offset[col];\r\n    uint ce = msc_column_offset[col + 1];\r\n\r\n    double dot_product = fgw_d[col] * v0[col];\r\n\t\tfor (uint i = cb; i < ce; ++i)\r\n    {\r\n      dot_product += msc_non_zero[i] * v0[msc_non_zero_row[i]];\r\n    }\r\n    v1[col] = dot_product;\r\n\r\n    sum[col] += fgw_w * dot_product;\r\n\t}\r\n}\r\n";

PS_StochTransientKernel::PS_StochTransientKernel
  ( cl::Device& cl_device_
  , cl::Context& cl_context_

  , cl_double* msc_non_zero
  , cl_uint* msc_non_zero_row
  , cl_uint* msc_col_offset
  , cl_uint msc_non_zero_size
  , cl_uint msc_dim

  , cl_double* fgw_d
  , cl_double* fgw_w_
  , cl_uint fgw_l
  )
  : cl_device_m(cl_device_)
  , cl_context_m(cl_context_)
  , cl_queue_m(cl_context(), cl_device(), 0, NULL)
  , cl_program_m(cl_context(), std::string(PS_StochTransientKernel::cl_kernel_source), true)
  , cl_kernel_m(cl_program(), "PS_StochTransientKernel")

  , msc_dim_m(msc_dim)
  , msc_non_zero_size_m(msc_non_zero_size)
  , fgw_w_m(fgw_w_)
  , fgw_l_m(fgw_l)
  , fgw_i_m(1)

  , cl_v0_m(cl_context(), CL_MEM_READ_WRITE, msc_dim_m * sizeof(cl_double))
  , cl_v1_m(cl_context(), CL_MEM_READ_WRITE, msc_dim_m * sizeof(cl_double))
  , cl_msc_non_zero_m(cl_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, msc_non_zero_size * sizeof(cl_double), msc_non_zero)
  , cl_msc_non_zero_row_m(cl_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, msc_non_zero_size * sizeof(cl_uint), msc_non_zero_row)
  , cl_msc_col_offset_m(cl_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (msc_dim + 1) * sizeof(cl_uint), msc_col_offset)
  , cl_fgw_d_m(cl_context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, msc_dim_m * sizeof(cl_double), fgw_d)
  , cl_sum_m(cl_context(), CL_MEM_READ_WRITE, msc_dim_m * sizeof(cl_double))
{
  cl_queue().enqueueFillBuffer(cl_sum_m, static_cast<cl_double>(0.0), 0, msc_dim_m * sizeof(cl_double));
  cl_queue().finish();

  cl_kernel().setArg(0, cl_msc_non_zero_m);
  cl_kernel().setArg(1, cl_msc_non_zero_row_m);
  cl_kernel().setArg(2, cl_msc_col_offset_m);
  cl_kernel().setArg(3, msc_dim_m);
  cl_kernel().setArg(4, cl_fgw_d_m);
  cl_kernel().setArg(6, cl_sum_m);
  cl_kernel().setArg(7, cl_v0_m);
  cl_kernel().setArg(8, cl_v1_m);
}

void PS_StochTransientKernel::run
  ( cl_double* vec_i
  , cl_double* vec_o
  , cl_uint times
  )
{
  size_t lws = 128;
  size_t gws = least_greater_multiple(lws, msc_dim_m);
 
  cl_queue().enqueueWriteBuffer(cl_v0_m, CL_TRUE, 0, msc_dim_m * sizeof(cl_double), vec_i);

  cl::Buffer& v0 = cl_v0_m;
  cl::Buffer& v1 = cl_v1_m;

  std::vector<cl::Event> ev_iter_exec(1);
  for (cl_uint ii = 0; ii < times; ++ii)
  {
    if (ii != 0)
    {
      cl::Event::waitForEvents(ev_iter_exec);
    }
    cl_kernel().setArg(5, fgw_w());
    cl_kernel().setArg(7, v0);
    cl_kernel().setArg(8, v1);
    std::swap(v0, v1);

    cl_int err = cl_queue().enqueueNDRangeKernel(cl_kernel(), cl::NullRange, gws, lws, NULL, &ev_iter_exec[0]); 

    ++fgw_i_m;
  }
  cl_queue().enqueueReadBuffer(v0, CL_TRUE, 0, msc_dim_m * sizeof(cl_double), vec_o, &ev_iter_exec);
}

void PS_StochTransientKernel::sum(cl_double* x)
{
  cl_queue().enqueueReadBuffer(cl_sum_m, CL_TRUE, 0, msc_dim_m * sizeof(cl_double), x);
}

