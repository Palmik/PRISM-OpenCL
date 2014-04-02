#ifndef PRISM_PS_FOX_GLYNN_OPENCL
#define PRISM_PS_FOX_GLYNN_OPENCL

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include "jnipointer.h"

typedef cl_double cl_real;

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
  );

class OpenCLKernel
{
  public:
    OpenCLKernel
      ( cl_device_id cl_device_
      , cl_context cl_context_
      , char const* source
      , char const* kernel_name
      );
    ~OpenCLKernel();

  protected:
    template <typename T>
    void cl_create_buffer(cl_mem& m, cl_mem_flags flags, size_t num, T* host_ptr = NULL)
    {
      cl_int err = 0;
      m = clCreateBuffer(cl_context_m, flags, num * sizeof(T), host_ptr, &err);
    }
    
    template <typename P>
    void cl_fill_buffer(cl_mem m, P pattern, size_t num, size_t offset = 0)
    {
      cl_int err = 0;
      cl_event ev;
      err = clEnqueueFillBuffer(cl_queue_m, m, &pattern, sizeof(P), offset, num * sizeof(P), 0, 0, &ev); 
      err = clWaitForEvents(1, &ev);
      clReleaseEvent(ev);
    }
    
    template <typename T>
    cl_int cl_write_buffer(cl_mem m, size_t num, T* host_ptr, size_t offset = 0)
    {
      return clEnqueueWriteBuffer(cl_queue_m, m, CL_TRUE, offset, num * sizeof(T), host_ptr, 0, 0, NULL); 
    }
    
    template <typename T>
    cl_int cl_read_buffer(cl_mem m, size_t num, T* host_ptr, size_t offset = 0)
    {
      return clEnqueueReadBuffer(cl_queue_m, m, CL_TRUE, offset, num * sizeof(T), host_ptr, 0, 0, NULL); 
    }

    template <typename T>
    cl_int cl_set_kernel_arg(cl_kernel k, cl_uint arg_index, T arg)
    {
      return clSetKernelArg(k, arg_index, sizeof(T), &arg);
    }
    
    cl_device_id cl_device_m;
    cl_context cl_context_m;
    cl_command_queue cl_queue_m;
    cl_program cl_program_m;
    cl_kernel cl_kernel_m;

};

static char const* PS_FoxGlynn_Source =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\r\n\r\ntypedef double real;\r\n\r\n__kernel void PS_FoxGlynn\r\n  ( const uint warp_size\r\n  , __global real const* fw_non_zero\r\n  , __global uint const* fw_non_zero_row\r\n  , __global uint const* fw_seg_offset\r\n  , const uint fw_ns\r\n  , const uint fw_ns_rem\r\n\r\n  , __global real const* fgw_d\r\n  , const real fgw_w\r\n  , __global real* sum\r\n\r\n  , __global real const* v0\r\n  , __global real* v1\r\n  )\r\n{\r\n  int col = get_group_id(0) * get_local_size(0) + get_local_id(0);\r\n  int seg_i = col / warp_size;\r\n  int off_i = get_local_id(0) % warp_size;\r\n\r\n  uint dim = (fw_ns - 1) * warp_size + fw_ns_rem;\r\n  if (col < dim)\r\n  {\r\n    real dot_product = fgw_d[col] * v0[col];\r\n    uint skip = (seg_i < fw_ns - 1) ? warp_size : fw_ns_rem;\r\n  \r\n    uint sb = fw_seg_offset[seg_i];\r\n    uint se = fw_seg_offset[seg_i + 1];\r\n    for (uint ii = sb + off_i; ii < se; ii += skip)\r\n    {\r\n      dot_product = fma(fw_non_zero[ii], v0[fw_non_zero_row[ii]], dot_product);\r\n    }\r\n    v1[col] = dot_product;\r\n\r\n    sum[col] = fma(fgw_w, dot_product, sum[col]);\r\n  }\r\n}\r\n\r\n__kernel void PS_FoxGlynn_Naive\r\n  ( __global real const* msc_non_zero\r\n  , __global uint const* msc_non_zero_row\r\n  , __global uint const* msc_column_offset\r\n  , const uint msc_dim\r\n\r\n  , __global real const* fgw_d\r\n  , const real fgw_w\r\n  , __global real* sum\r\n\r\n  , __global real const* v0\r\n  , __global real* v1\r\n  )\r\n{\r\n  int col = get_global_id(0);\r\n  if (col < msc_dim)\r\n  {\r\n    uint cb = msc_column_offset[col];\r\n    uint ce = msc_column_offset[col + 1];\r\n\r\n    real dot_product = fgw_d[col] * v0[col];\r\n    for (uint i = cb; i < ce; ++i)\r\n    {\r\n      dot_product += msc_non_zero[i] * v0[msc_non_zero_row[i]];\r\n    }\r\n    v1[col] = dot_product;\r\n\r\n    sum[col] += fgw_w * dot_product;\r\n  }\r\n}\r\n\r\n";

class PS_FoxGlynn_OpenCLKernel : public OpenCLKernel
{
  public:
    PS_FoxGlynn_OpenCLKernel
      ( cl_device_id cl_device_
      , cl_context cl_context_

      , cl_real* msc_non_zero
      , cl_uint* msc_non_zero_row
      , cl_uint* msc_col_offset
      , cl_uint msc_non_zero_size
      , cl_uint msc_dim

      , cl_real* fgw_ds
      , cl_real* fgw_ws
      , cl_uint fgw_l
      );
    ~PS_FoxGlynn_OpenCLKernel();

    void run(cl_real* vec_i, cl_real* vec_o, cl_uint times = 1);
    void sum(cl_real* x);

  private:
    cl_real fgw_w() { return ((fgw_i_m < fgw_l_m) ? 0.0 : fgw_w_m[fgw_i_m - fgw_l_m]); }

    cl_uint dim_m;
    cl_uint msc_non_zero_size_m;
    
    cl_real* fgw_w_m; // The FGW weights.
    cl_uint fgw_l_m; // The FGW "left" parameter.
    cl_uint fgw_i_m; // The FGW iteration counter.
  
    cl_mem cl_v0_m;
    cl_mem cl_v1_m;
    cl_mem cl_fw_non_zero_m;
    cl_mem cl_fw_non_zero_row_m;
    cl_mem cl_fw_seg_offset_m;

    cl_mem cl_fgw_d_m;
    cl_mem cl_sum_m;

    size_t lws_m;
    size_t gws_m;
};

#ifdef CL_FOX_GLYNN_NAIVE
class PS_FoxGlynn_OpenCLKernelNaive : public OpenCLKernel
{
  public:
    PS_FoxGlynn_OpenCLKernelNaive
      ( cl_device_id cl_device_
      , cl_context cl_context_

      , cl_real* msc_non_zero
      , cl_uint* msc_non_zero_row
      , cl_uint* msc_col_offset
      , cl_uint msc_non_zero_size
      , cl_uint msc_dim

      , cl_real* fgw_ds
      , cl_real* fgw_ws
      , cl_uint fgw_l
      );
    ~PS_FoxGlynn_OpenCLKernelNaive();

    void run(cl_real* vec_i, cl_real* vec_o, cl_uint times = 1);
    void sum(cl_real* x);

  private:
    cl_real fgw_w() { return ((fgw_i_m < fgw_l_m) ? 0.0 : fgw_w_m[fgw_i_m - fgw_l_m]); }

    cl_uint dim_m;
    cl_uint msc_non_zero_size_m;
    
    cl_real* fgw_w_m; // The FGW weights.
    cl_uint fgw_l_m; // The FGW "left" parameter.
    cl_uint fgw_i_m; // The FGW iteration counter.
  
    cl_mem cl_v0_m;
    cl_mem cl_v1_m;
    cl_mem cl_msc_non_zero_m;
    cl_mem cl_msc_non_zero_row_m;
    cl_mem cl_msc_col_offset_m;

    cl_mem cl_fgw_d_m;
    cl_mem cl_sum_m;

    size_t lws_m;
    size_t gws_m;
};
#endif // CL_FOX_GLYNN_NAIVE

#endif // PRISM_PS_FOX_GLYNN_OPENCL
