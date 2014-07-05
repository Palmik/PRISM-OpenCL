#ifndef FOXGLYNN_OPENCL_KERNEL
#define FOXGLYNN_OPENCL_KERNEL

#include <CL/cl.h>
#include "OpenCLUtil.h"

template <typename FGCore>
class FGKernel 
{
  public:
    FGKernel
      ( cl_device_id cl_device_
      , cl_context cl_context_
      
      , MatrixCS matrix
      , cl_real* fgw_ds // fgw diagonals
      , cl_real* fgw_ws // fgw weights
      , cl_uint fgw_l 
      , cl_uint fgw_r

      , cl_real unif
      , bool is_cumul_reward
      )
      : cl_device_m(cl_device_)
      , cl_context_m(cl_context_)

      , core_m(cl_device_, cl_context_, matrix, fgw_ds)
      , matrix_m(matrix)
      , fgw_ws_m(fgw_ws)
      , fgw_l_m(fgw_l)
      , fgw_r_m(fgw_r)
      , fgw_i_m(1)

      , unif_m(unif)
      , is_cumul_reward_m(is_cumul_reward)
      {
        cl_int err = 0;

        cl_kernel_m = core_m.get_kernel();
        cl_queue_m = clCreateCommandQueue(cl_context_m, cl_device_m, 0, &err);
        cl_create_buffer<cl_real>(cl_context_m, cl_v0_m, CL_MEM_READ_WRITE, matrix_m.n);
        cl_create_buffer<cl_real>(cl_context_m, cl_v1_m, CL_MEM_READ_WRITE, matrix_m.n);
        cl_create_buffer<cl_real>(cl_context_m, cl_sum_m, CL_MEM_READ_WRITE, matrix_m.n);
        cl_fill_buffer<cl_real>(cl_queue_m, cl_sum_m, 0, matrix_m.n);
        core_m.set_sum_mem(cl_sum_m);
      }
   
    ~FGKernel()
    {
      clReleaseMemObject(cl_sum_m);
      clReleaseMemObject(cl_v1_m);
      clReleaseMemObject(cl_v0_m);
      clReleaseCommandQueue(cl_queue_m);
    }

    void sum(cl_real* x)
    {
      cl_read_buffer<cl_real>(cl_queue_m, cl_sum_m, matrix_m.n, x);
    }

    void run
      ( cl_real* vec_i
      , cl_real* vec_o
      , cl_uint times
      )
    {
      cl_int err = 0;
      size_t lws_m = 256;
      size_t gws_m = least_greater_multiple(lws_m, (size_t)matrix_m.n);
      cl_write_buffer<cl_real>(cl_queue_m, cl_v0_m, matrix_m.n, vec_i);

      cl_mem& v0 = cl_v0_m;
      cl_mem& v1 = cl_v1_m;
      cl_event ev_iter_exec = NULL;
      for (cl_uint ii = 0; ii < times; ++ii)
      {
        if (ii != 0)
        {
          err = clWaitForEvents(1, &ev_iter_exec); CLERR();
        }
        core_m.set_fgw_w(fgw_w());
        core_m.set_veci_mem(v0);
        core_m.set_veco_mem(v1);
        std::swap(v0, v1);

        err = clEnqueueNDRangeKernel(cl_queue_m, cl_kernel_m, 1, NULL, &gws_m, &lws_m, 0, NULL, &ev_iter_exec); CLERR();

        ++fgw_i_m;
      }
      err = clWaitForEvents(1, &ev_iter_exec); CLERR();
      cl_read_buffer<cl_real>(cl_queue_m, v0, matrix_m.n, vec_o);
      cl_read_buffer<cl_real>(cl_queue_m, v1, matrix_m.n, vec_i);
      err = clReleaseEvent(ev_iter_exec); CLERR();
    }
    
  private:
    cl_real fgw_w()
    {
      if (fgw_i_m < fgw_l_m)
      {
        if (is_cumul_reward_m) { return 1.0 / unif_m; }
        else { return 0.0; }
      }
      else
      {
        return fgw_ws_m[fgw_i_m - fgw_l_m];
      }
    }

    cl_device_id cl_device_m;
    cl_context cl_context_m;
    cl_command_queue cl_queue_m;
    cl_kernel cl_kernel_m;

    FGCore core_m;
    cl_mem cl_v0_m;
    cl_mem cl_v1_m;
    cl_mem cl_sum_m;
    
    MatrixCS matrix_m;
    cl_real* fgw_ws_m;
    cl_uint fgw_l_m;
    cl_uint fgw_r_m;
    cl_uint fgw_i_m;

    bool is_cumul_reward_m;
    cl_real unif_m;
};

#endif // FOXGLYNN_OPENCL_KERNEL
