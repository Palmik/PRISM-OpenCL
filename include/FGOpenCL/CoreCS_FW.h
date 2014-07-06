#ifndef FOXGLYNN_OPENCL_CORE_CS_FW
#define FOXGLYNN_OPENCL_CORE_CS_FW

#include <iostream>

#include <CL/cl.h>
#include "OpenCLUtil.h"

#include "FGOpenCL/KernelSource.h"

class FGCoreCS_FW : public OpenCLKernel
{
  public:
    FGCoreCS_FW(cl_device_id cl_device_, cl_context cl_context_, MatrixCS matrix, cl_real* fgw_ds)
      : OpenCLKernel(cl_device_, cl_context_, FGOpenCLSrc, "FGCoreCS_FW") 
    {
      cl_int err = 0;

      cl_uint warp_size = cl_warp_size(cl_device_, cl_kernel_m);
      std::cerr << "Warp size: " << warp_size << std::endl;

      cl_uint K = 0; // The maximum number of non-zero values in a single row (y).
      for (cl_uint ii = 0; ii < matrix.n; )
      {
        K = std::max(K, matrix.non_zero_ys_offset[ii + 1] - matrix.non_zero_ys_offset[ii]);
      }
      
      std::vector<cl_real> ell_data;
      std::vector<cl_real> ell_data_xs;
      for (cl_uint ii = 0; ii < matrix.n; )
      {
        cl_uint yb = matrix.non_zero_ys_offset[ii];
        cl_uint ye = matrix.non_zero_ys_offset[ii + 1];

        for (size_t jj = yb; jj < ye; ++jj)
        {
          ell_data.push_back(matrix.non_zero[jj]);
          ell_data_xs.push_back(matrix.non_zero_xs[jj]);
        }

        for (size_t jj = ye - yb; jj < K; ++jj)
        {
          ell_data.push_back(0);
          ell_data_xs.push_back(0);
        }
      }

      // Create the buffers.
      cl_create_buffer<cl_real>(cl_fw_non_zero_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fw_non_zero.size(), fw_non_zero.data());
      cl_create_buffer<cl_uint>(cl_fw_non_zero_row_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fw_non_zero_row.size(), fw_non_zero_row.data());
      cl_create_buffer<cl_uint>(cl_fw_seg_offset_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fw_seg_offset.size(), fw_seg_offset.data());

      cl_create_buffer<cl_real>(cl_fgw_d_m, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrix.n, fgw_ds);

      cl_uint fw_ns = (matrix.n + (warp_size - 1)) / warp_size;
      cl_uint fw_ns_rem = (matrix.n % warp_size) ? matrix.n % warp_size : warp_size;

      cl_set_arg(0, warp_size);
      cl_set_arg(1, cl_fw_non_zero_m);
      cl_set_arg(2, cl_fw_non_zero_row_m);
      cl_set_arg(3, cl_fw_seg_offset_m);
      cl_set_arg(4, fw_ns);
      cl_set_arg(5, fw_ns_rem);
      cl_set_arg(6, cl_fgw_d_m);
      // fgw_w
    }

    ~FGCoreCS_FW()
    {
      clReleaseMemObject(cl_fgw_d_m);
      clReleaseMemObject(cl_fw_seg_offset_m);
      clReleaseMemObject(cl_fw_non_zero_row_m);
      clReleaseMemObject(cl_fw_non_zero_m);
    }

    cl_kernel get_kernel() { return cl_kernel_m; }
    void set_fgw_w(cl_real w) { cl_set_arg(7, w); } 
    void set_veci_mem(cl_mem m) { cl_set_arg(9, m); }
    void set_veco_mem(cl_mem m) { cl_set_arg(10, m); }
    void set_sum_mem(cl_mem m) { cl_set_arg(8, m); }

  private:
    cl_mem cl_fw_non_zero_m;
    cl_mem cl_fw_non_zero_row_m;
    cl_mem cl_fw_seg_offset_m;

    cl_mem cl_fgw_d_m;
    
    cl_mem cl_veci_m;
    cl_mem cl_veco_m;
    cl_mem cl_sum_m;

    size_t lws_m;
    size_t gws_m;
};

#endif // FOXGLYNN_OPENCL_CORE_CS_FW
