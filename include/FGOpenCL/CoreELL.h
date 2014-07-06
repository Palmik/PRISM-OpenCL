#ifndef FOXGLYNN_OPENCL_CORE_ELL
#define FOXGLYNN_OPENCL_CORE_ELL

#include <CL/cl.h>
#include "OpenCLUtil.h"

#include "FGOpenCL/KernelSource.h"

class FGCoreELL : public OpenCLKernel
{
  public:
    FGCoreELL(cl_device_id cl_device_, cl_context cl_context_, MatrixELL matrix, cl_real* fgw_ds)
      : OpenCLKernel(cl_device_, cl_context_, FGOpenCLSrc, "FGCoreELL") 
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
    MatrixELL matrix_m;
   
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

#endif // FOXGLYNN_OPENCL_CORE_ELL
