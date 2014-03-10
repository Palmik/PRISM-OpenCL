#ifndef PRISM_PS_STOCH_TRANSIENT_KERNEL
#define PRISM_PS_STOCH_TRANSIENT_KERNEL

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

typedef cl_double cl_real;

class PS_FGKernel
{
  public:
    PS_FGKernel
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
    ~PS_FGKernel();

    void run(cl_real* vec_i, cl_real* vec_o, cl_uint times = 1);
    void sum(cl_real* x);

  private:
    static char const* cl_program_source;

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
    
    cl_real fgw_w() { return ((fgw_i_m < fgw_l_m) ? 0.0 : fgw_w_m[fgw_i_m - fgw_l_m]); }

    cl_device_id cl_device_m;
    cl_context cl_context_m;
    cl_command_queue cl_queue_m;
    cl_program cl_program_m;
    cl_kernel cl_kernel_m;

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

#endif // PRISM_PS_STOCH_TRANSIENT_KERNEL
