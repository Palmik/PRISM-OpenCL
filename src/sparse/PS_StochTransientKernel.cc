#include "PS_StochTransientKernel.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <iostream>

#include <CL/cl.h>

unsigned long int least_greater_multiple(unsigned long int a, unsigned long int min)
{
  unsigned long int r = a;
  while (r < min) { r *= a; }
  return r;
}

PS_StochTransientKernel::PS_StochTransientKernel
  ( cl_device_id device_id, cl_context context
  , cl_double* msc_non_zero
  , cl_uint* msc_non_zero_row
  , cl_uint* msc_col_offset
  , cl_uint msc_non_zero_size
  , cl_uint msc_dim

  , cl_double* fgw_d
  )
  : device_id_m(device_id)
  , context_m(context)
  , dim_m(msc_dim)
{
  cl_int err = 0;

  char const* source = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\r\n\r\n__kernel void PS_StochTransientKernel\r\n  ( __global double const* msc_non_zero\r\n  , __global uint const* msc_non_zero_row\r\n  , __global uint const* msc_column_offset\r\n  , const uint msc_dim\r\n\r\n  , __global double const* fgw_d\r\n\r\n  , __global double* v1\r\n  , __global double const* v0\r\n  )\r\n{\r\n\tint col = get_global_id(0);\r\n\tif (col < msc_dim)\r\n\t{\r\n    uint cb = msc_column_offset[col];\r\n    uint ce = msc_column_offset[col + 1];\r\n\r\n    double dot_product = fgw_d[col] * v0[col];\r\n\t\tfor (uint i = cb; i < ce; ++i)\r\n    {\r\n      dot_product += msc_non_zero[i] * v0[msc_non_zero_row[i]];\r\n    }\r\n    v1[col] = dot_product;\r\n\t}\r\n}\r\n";

  program_m = clCreateProgramWithSource(context, 1, &source, NULL, &err);
  cl_error_check(err, "clCreateProgramWithSource");
  err = clBuildProgram(program_m, 1, &device_id, NULL, NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t log_size;
    clGetProgramBuildInfo(program_m, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char*) malloc(log_size);
    clGetProgramBuildInfo(program_m, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    std::printf("%s\n", log);
    free(log);
  }
  else
  {
    cl_error_check(err, "clBuildProgram");
  }

  kernel_m = clCreateKernel(program_m, "PS_StochTransientKernel", &err);
  cl_error_check(err, "clCreateKernel");
  
  vec_out_cl_m = clCreateBuffer
    ( context_m
    , CL_MEM_WRITE_ONLY
    , dim_m * sizeof(cl_double)
    , NULL
    , &err
    );
  cl_error_check(err, "clCreateBuffer(vec_out)");
  
  non_zero_cl_m = clCreateBuffer
    ( context_m
    , CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
    , msc_non_zero_size * sizeof(cl_double)
    , msc_non_zero
    , &err
    );
  cl_error_check(err, "clCreateBuffer(non_zero)");

  non_zero_row_cl_m = clCreateBuffer
    ( context_m
    , CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
    , msc_non_zero_size * sizeof(cl_uint)
    , msc_non_zero_row
    , &err
    );
  cl_error_check(err, "clCreateBuffer(non_zero_row)");
  
  column_offset_cl_m = clCreateBuffer
    ( context_m
    , CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
    , (dim_m + 1) * sizeof(cl_uint)
    , msc_col_offset
    , &err
    );
  cl_error_check(err, "clCreateBuffer(column_offset)");
  
  fgw_d_cl_m = clCreateBuffer
    ( context_m
    , CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
    , dim_m * sizeof(cl_double)
    , fgw_d
    , &err
    );
  cl_error_check(err, "clCreateBuffer(fgw_d)");
  
  err = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), &non_zero_cl_m);
  cl_error_check(err, "clSetKernelArg(non_zero)");
  err = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), &non_zero_row_cl_m);
  cl_error_check(err, "clSetKernelArg(non_zero_row)");
  err = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), &column_offset_cl_m);
  cl_error_check(err, "clSetKernelArg(col_offset)");
  err = clSetKernelArg(kernel_m, 3, sizeof(cl_uint), &dim_m);
  cl_error_check(err, "clSetKernelArg(dim)");
  err = clSetKernelArg(kernel_m, 4, sizeof(cl_mem), &fgw_d_cl_m);
  cl_error_check(err, "clSetKernelArg(fgw_d)");
  err = clSetKernelArg(kernel_m, 5, sizeof(cl_mem), &vec_out_cl_m);
  cl_error_check(err, "clSetKernelArg(vec_out)");
}

PS_StochTransientKernel::~PS_StochTransientKernel()
{
  cl_uint err = 0;
 
  err = clReleaseKernel(kernel_m);
  cl_error_check(err, "clReleaseKernel");

  err = clReleaseMemObject(fgw_d_cl_m);
  cl_error_check(err, "clReleaseMemObject(fgw_d)");

  err = clReleaseMemObject(column_offset_cl_m);
  cl_error_check(err, "clReleaseMemObject(column_offset)");

  err = clReleaseMemObject(non_zero_row_cl_m);
  cl_error_check(err, "clReleaseMemObject(non_zero_row)");

  err = clReleaseMemObject(non_zero_cl_m);
  cl_error_check(err, "clReleaseMemObject(non_zero)");

  err = clReleaseMemObject(vec_out_cl_m);
  cl_error_check(err, "clReleaseMemObject(vec_out_cl_m)");

}

void PS_StochTransientKernel::run
  ( cl_command_queue queue
  , cl_double* vec_in
  , cl_double* vec_out
  , cl_uint times
  )
{
  cl_int err = 0; 
  
  cl_mem vec_in_cl_m = clCreateBuffer
    ( context_m
    , CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
    , dim_m * sizeof(cl_double)
    , vec_in
    , &err
    );
  cl_error_check(err, "clCreateBuffer(vec_in)");
  
  err = clSetKernelArg(kernel_m, 6, sizeof(cl_mem), &vec_in_cl_m);
  cl_error_check(err, "clSetKernelArg(vec_in)");

  size_t local_work_size = 128;
  size_t global_work_size = least_greater_multiple(local_work_size, dim_m);
  
  cl_event prev_run_event;
  cl_event prev_copy_event;

  err = clEnqueueNDRangeKernel(queue, kernel_m, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &prev_run_event);
  cl_error_check(err, "clEnqueueNDRangeKernel");

  for (cl_uint i = 1; i < times; ++i)
  {
    err = clEnqueueCopyBuffer(queue, vec_out_cl_m, vec_in_cl_m, 0, 0, dim_m * sizeof(cl_double), 1, &prev_run_event, &prev_copy_event);
    cl_error_check(err, "clEnqueueCopyBuffer"); 
    err = clEnqueueNDRangeKernel(queue, kernel_m, 1, NULL, &global_work_size, &local_work_size, 1, &prev_copy_event, &prev_run_event);
    cl_error_check(err, "clEnqueueNDRangeKernel");
  }
  
  err = clEnqueueReadBuffer(queue, vec_out_cl_m, CL_TRUE, 0, dim_m * sizeof(cl_double), vec_out, 1, &prev_run_event, NULL);
  cl_error_check(err, "clEnqueueCopyBuffer");

  err = clReleaseMemObject(vec_in_cl_m); 
  cl_error_check(err, "clReleaseMemObject(vec_in)");
}

