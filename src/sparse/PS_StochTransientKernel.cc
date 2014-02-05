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

  , cl_double* fgw_ds
  , cl_double* fgw_ws
  , cl_uint fgw_l
  )
  : device_id_m(device_id)
  , context_m(context)
  , zero_m(0.0)
  , fgw_ws_m(fgw_ws)
  , fgw_l_m(fgw_l)
  , fgw_iteration_m(1)
  , dim_m(msc_dim)
{
  cl_int err = 0;

  char const* source = 
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\r\n\r\n__kernel void PS_StochTransientKernel\r\n  ( __global double const* msc_non_zero\r\n  , __global uint const* msc_non_zero_row\r\n  , __global uint const* msc_column_offset\r\n  , const uint msc_dim\r\n\r\n  , __global double const* fgw_d\r\n  , __global double const* fgw_w\r\n  , __global double* sum\r\n\r\n  , __global double const* v0\r\n  , __global double* v1\r\n  )\r\n{\r\n\tint col = get_global_id(0);\r\n\tif (col < msc_dim)\r\n\t{\r\n    uint cb = msc_column_offset[col];\r\n    uint ce = msc_column_offset[col + 1];\r\n\r\n    double dot_product = fgw_d[col] * v0[col];\r\n\t\tfor (uint i = cb; i < ce; ++i)\r\n    {\r\n      dot_product += msc_non_zero[i] * v0[msc_non_zero_row[i]];\r\n    }\r\n    v1[col] = dot_product;\r\n\r\n    sum[col] += fgw_w[0] * dot_product;\r\n\t}\r\n}\r\n";

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
 
  cl_command_queue queue = clCreateCommandQueue(context_m, device_id_m, 0, &err);
  cl_error_check(err, "clCreateCommandQueue");

  vec_out_cl_m = cl_create_buffer(queue, CL_MEM_WRITE_ONLY, dim_m * sizeof(cl_double)); 
  
  non_zero_cl_m = cl_create_buffer(queue, CL_MEM_READ_ONLY, msc_non_zero_size * sizeof(cl_double), msc_non_zero);
  non_zero_row_cl_m = cl_create_buffer(queue, CL_MEM_READ_ONLY, msc_non_zero_size * sizeof(cl_uint), msc_non_zero_row);
  column_offset_cl_m = cl_create_buffer(queue, CL_MEM_READ_ONLY, (dim_m + 1) * sizeof(cl_uint), msc_col_offset);
  fgw_ds_cl_m = cl_create_buffer(queue, CL_MEM_READ_ONLY, dim_m * sizeof(cl_double), fgw_ds);
  fgw_w_cl_m = cl_create_buffer(queue, CL_MEM_READ_ONLY, sizeof(cl_double));
  sum_cl_m = cl_create_buffer_with_pattern(queue, CL_MEM_READ_ONLY, dim_m * sizeof(cl_double), sizeof(cl_double), &zero_m); 

  clFinish(queue);
  
  err = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), &non_zero_cl_m);
  cl_error_check(err, "clSetKernelArg(non_zero)");
  err = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), &non_zero_row_cl_m);
  cl_error_check(err, "clSetKernelArg(non_zero_row)");
  err = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), &column_offset_cl_m);
  cl_error_check(err, "clSetKernelArg(col_offset)");
  err = clSetKernelArg(kernel_m, 3, sizeof(cl_uint), &dim_m);
  cl_error_check(err, "clSetKernelArg(dim)");
  err = clSetKernelArg(kernel_m, 4, sizeof(cl_mem), &fgw_ds_cl_m);
  cl_error_check(err, "clSetKernelArg(fgw_ds)");
  err = clSetKernelArg(kernel_m, 5, sizeof(cl_mem), &fgw_w_cl_m);
  cl_error_check(err, "clSetKernelArg(fgw_w)");
  err = clSetKernelArg(kernel_m, 6, sizeof(cl_mem), &sum_cl_m);
  cl_error_check(err, "clSetKernelArg(sum)");
  err = clSetKernelArg(kernel_m, 8, sizeof(cl_mem), &vec_out_cl_m);
  cl_error_check(err, "clSetKernelArg(vec_out)");
}

cl_mem PS_StochTransientKernel::cl_create_buffer
  ( cl_command_queue queue
  , cl_mem_flags mem_flags
  , size_t size
  , void* host_ptr
  , cl_bool blocking
  )
{
  cl_int err = 0;

  cl_mem res = clCreateBuffer(context_m, mem_flags, size, NULL, &err);
  cl_error_check(err, "clCreateBuffer(cl_create_buffer)");

  if (host_ptr != NULL)
  {
    err = clEnqueueWriteBuffer(queue, res, blocking, 0, size, host_ptr, 0, NULL, NULL);
    cl_error_check(err, "clEnqueueWriteBuffer(cl_create_buffer)");
  }

  return res;
}

cl_mem PS_StochTransientKernel::cl_create_buffer_with_pattern
  ( cl_command_queue queue
  , cl_mem_flags mem_flags
  , size_t size
  , size_t pattern_size
  , void* pattern
  )
{
  cl_int err = 0;

  cl_mem res = clCreateBuffer(context_m, mem_flags, size, NULL, &err);
  cl_error_check(err, "clCreateBuffer(cl_create_buffer_with_pattern)");

  err = clEnqueueFillBuffer(queue, res, pattern, pattern_size, 0, size, 0, NULL, NULL);
  cl_error_check(err, "clEnqueueFillBuffer(cl_create_buffer_with_pattern)");

  return res;
}

PS_StochTransientKernel::~PS_StochTransientKernel()
{
  cl_uint err = 0;
 
  err = clReleaseKernel(kernel_m);
  cl_error_check(err, "clReleaseKernel");

  err = clReleaseMemObject(fgw_ds_cl_m);
  cl_error_check(err, "clReleaseMemObject(fgw_ds)");

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
  
  err = clSetKernelArg(kernel_m, 7, sizeof(cl_mem), &vec_in_cl_m);
  cl_error_check(err, "clSetKernelArg(vec_in)");

  size_t local_work_size = 128;
  size_t global_work_size = least_greater_multiple(local_work_size, dim_m);
  
  cl_event ev_exec = NULL; // Execute the kernel.
  cl_event ev_fgww = NULL; // Set the new fgw weight.

  cl_double* fgww = (fgw_iteration_m < fgw_l_m) ? &zero_m : &fgw_ws_m[fgw_iteration_m - fgw_l_m];
  err = clEnqueueWriteBuffer(queue, fgw_w_cl_m, CL_FALSE, 0, sizeof(cl_double), fgww, 0, NULL, &ev_fgww); 
  cl_error_check(err, "clEnqueueWriteBuffer(fgw_w_cl_m)");
  
  err = clEnqueueNDRangeKernel(queue, kernel_m, 1, NULL, &global_work_size, &local_work_size, 1, &ev_fgww, &ev_exec);
  cl_error_check(err, "clEnqueueNDRangeKernel");
  ++fgw_iteration_m;

  for (cl_uint i = 1; i < times; ++i)
  {
    cl_double* fgww = (fgw_iteration_m < fgw_l_m) ? &zero_m : &fgw_ws_m[fgw_iteration_m - fgw_l_m];
    cl_event ev_copy = NULL; // Make the resulting vector the input vector.

    err = clEnqueueCopyBuffer(queue, vec_out_cl_m, vec_in_cl_m, 0, 0, dim_m * sizeof(cl_double), 1, &ev_exec, &ev_copy);
    cl_error_check(err, "clEnqueueCopyBuffer"); 
  
    clReleaseEvent(ev_fgww);
    err = clEnqueueWriteBuffer(queue, fgw_w_cl_m, CL_FALSE, 0, sizeof(cl_double), fgww, 1, &ev_copy, &ev_fgww); 
    cl_error_check(err, "clEnqueueWriteBuffer(fgw_w_cl_m)");
    
    clReleaseEvent(ev_exec);
    err = clEnqueueNDRangeKernel(queue, kernel_m, 1, NULL, &global_work_size, &local_work_size, 1, &ev_fgww, &ev_exec);
    cl_error_check(err, "clEnqueueNDRangeKernel");
    
    clReleaseEvent(ev_copy);
    
    ++fgw_iteration_m;
  }
  
  err = clEnqueueReadBuffer(queue, vec_out_cl_m, CL_TRUE, 0, dim_m * sizeof(cl_double), vec_out, 1, &ev_exec, NULL);
  cl_error_check(err, "clEnqueueCopyBuffer");

  clReleaseEvent(ev_exec);

  err = clReleaseMemObject(vec_in_cl_m); 
  cl_error_check(err, "clReleaseMemObject(vec_in)");
}

void PS_StochTransientKernel::sum(cl_command_queue queue, cl_double* x)
{
  cl_int err = 0;

  err = clEnqueueReadBuffer(queue, sum_cl_m, CL_TRUE, 0, dim_m * sizeof(cl_double), x, 0, NULL, NULL);
  cl_error_check(err, "clEnqueueCopyBuffer");
}

