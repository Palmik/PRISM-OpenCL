#ifndef PRISM_PS_STOCH_TRANSIENT_KERNEL
#define PRISM_PS_STOCH_TRANSIENT_KERNEL

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class PS_StochTransientKernel
{
  public:
    PS_StochTransientKernel
      ( cl::Device&  cl_device
      , cl::Context& cl_context

      , cl_double* msc_non_zero
      , cl_uint* msc_non_zero_row
      , cl_uint* msc_col_offset
      , cl_uint msc_non_zero_size
      , cl_uint msc_dim

      , cl_double* fgw_ds
      , cl_double* fgw_ws
      , cl_uint fgw_l
      );

    void run(cl_double* vec_i, cl_double* vec_o, cl_uint times = 1);
    void sum(cl_double* x);

  private:
    static char const* cl_kernel_source;

    cl::Device& cl_device() { return cl_device_m; }
    cl::Context& cl_context() { return cl_context_m; }
    cl::CommandQueue& cl_queue() { return cl_queue_m; }
    cl::Program& cl_program() { return cl_program_m; }
    cl::Kernel& cl_kernel() { return cl_kernel_m; }

    cl_double fgw_w() { return ((fgw_i_m < fgw_l_m) ? 0.0 : fgw_w_m[fgw_i_m - fgw_l_m]); }

    cl::Device& cl_device_m;
    cl::Context& cl_context_m;
    cl::CommandQueue cl_queue_m;
    cl::Program cl_program_m;
    cl::Kernel cl_kernel_m;

    cl_uint msc_dim_m;
    cl_uint msc_non_zero_size_m;
    
    cl_double* fgw_w_m; // The FGW weights.
    cl_uint fgw_l_m; // The FGW "left" parameter.
    cl_uint fgw_i_m; // The FGW iteration counter.
  
    cl::Buffer cl_v0_m;
    cl::Buffer cl_v1_m;
    cl::Buffer cl_msc_non_zero_m;
    cl::Buffer cl_msc_non_zero_row_m;
    cl::Buffer cl_msc_col_offset_m;

    cl::Buffer cl_fgw_d_m;
    cl::Buffer cl_sum_m;

};

#endif // PRISM_PS_STOCH_TRANSIENT_KERNEL
