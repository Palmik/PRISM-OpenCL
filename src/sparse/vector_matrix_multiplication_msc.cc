#include "vector_matrix_multiplication_msc.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>

#include <CL/cl.h>

std::string read_file(char const* file_name)
{
  std::ifstream file(file_name);
  std::string file_str;

  file.seekg(0, std::ios::end);
  file_str.reserve(file.tellg());
  file.seekg(0, std::ios::beg);

  file_str.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  return file_str;
}

unsigned long int least_greater_multiple(unsigned long int a, unsigned long int min)
{
  unsigned long int r = a;
  while (r < min) { r *= a; }
  return r;
}

vector_matrix_multiplication_msc::vector_matrix_multiplication_msc
  ( cl_device_id device_id, cl_context context
  , float* msc_non_zero
  , cl_uint* msc_non_zero_row
  , cl_uint* msc_col_offset
  , cl_uint msc_non_zero_size
  , cl_uint msc_dim
  )
  : device_id_m(device_id)
  , context_m(context)
  , dim_m(msc_dim)
{
  cl_int err = 0;

  std::string program_src = read_file("vector_matrix_multiplication_msc.cl");
  const char* program_src_data = program_src.data();
  size_t program_src_size = program_src.size();
  program_m = clCreateProgramWithSource(context, 1, &program_src_data, &program_src_size, &err);
  cl_error_check(err, "clCreateProgramWithSource");
  err = clBuildProgram(program_m, 1, &device_id, NULL, NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t log_size;
    clGetProgramBuildInfo(program_m, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char*) malloc(log_size);
    clGetProgramBuildInfo(program_m, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    std::printf("%s\n", log);
  }
  else
  {
    cl_error_check(err, "clBuildProgram");
  }

  kernel_m = clCreateKernel(program_m, "vector_matrix_multiplication_msc", &err);
  cl_error_check(err, "clCreateKernel");
  
  vec_out_cl_m = clCreateBuffer
    ( context_m
    , CL_MEM_WRITE_ONLY
    , dim_m * sizeof(float)
    , NULL
    , &err
    );
  cl_error_check(err, "clCreateBuffer(vec_out)");
  
  non_zero_cl_m = clCreateBuffer
    ( context_m
    , CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
    , msc_non_zero_size * sizeof(float)
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
  
  err = clSetKernelArg(kernel_m, 0, sizeof(cl_mem), &non_zero_cl_m);
  cl_error_check(err, "clSetKernelArg(non_zero)");
  err = clSetKernelArg(kernel_m, 1, sizeof(cl_mem), &non_zero_row_cl_m);
  cl_error_check(err, "clSetKernelArg(non_zero_row)");
  err = clSetKernelArg(kernel_m, 2, sizeof(cl_mem), &column_offset_cl_m);
  cl_error_check(err, "clSetKernelArg(col_offset)");
  err = clSetKernelArg(kernel_m, 3, sizeof(cl_uint), &dim_m);
  cl_error_check(err, "clSetKernelArg(dim)");
  err = clSetKernelArg(kernel_m, 4, sizeof(cl_mem), &vec_out_cl_m);
  cl_error_check(err, "clSetKernelArg(vec_out)");
}

vector_matrix_multiplication_msc::~vector_matrix_multiplication_msc()
{
}

void vector_matrix_multiplication_msc::run
  ( cl_command_queue queue
  , float* vec_in
  , float* vec_out
  , cl_uint times
  )
{
  cl_int err = 0; 
  
  cl_mem vec_in_cl_m = clCreateBuffer
    ( context_m
    , CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
    , dim_m * sizeof(float)
    , vec_in
    , &err
    );
  cl_error_check(err, "clCreateBuffer(vec_in)");
  
  err = clSetKernelArg(kernel_m, 5, sizeof(cl_mem), &vec_in_cl_m);
  cl_error_check(err, "clSetKernelArg(vec_in)");

  size_t local_work_size = 128;
  size_t global_work_size = least_greater_multiple(local_work_size, dim_m);
  
  cl_event prev_run_event;
  cl_event prev_copy_event;

  err = clEnqueueNDRangeKernel(queue, kernel_m, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &prev_run_event);
  cl_error_check(err, "clEnqueueNDRangeKernel");

  for (cl_uint i = 1; i < times; ++i)
  {
    err = clEnqueueCopyBuffer(queue, vec_out_cl_m, vec_in_cl_m, 0, 0, dim_m * sizeof(float), 1, &prev_run_event, &prev_copy_event);
    cl_error_check(err, "clEnqueueCopyBuffer"); 
    err = clEnqueueNDRangeKernel(queue, kernel_m, 1, NULL, &global_work_size, &local_work_size, 1, &prev_copy_event, &prev_run_event);
    cl_error_check(err, "clEnqueueNDRangeKernel");
  }
  
  err = clEnqueueReadBuffer(queue, vec_out_cl_m, CL_TRUE, 0, dim_m * sizeof(float), vec_out, 1, &prev_run_event, NULL);
  cl_error_check(err, "clEnqueueCopyBuffer"); 
}

