#ifndef PRISM_PS_STOCH_TRANSIENT_KERNEL
#define PRISM_PS_STOCH_TRANSIENT_KERNEL

#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>

static inline void cl_error_check(cl_int err, char const* msg = "")
{
  if (err != CL_SUCCESS)
  {
    std::printf("OpenCL Error: %s (%d)\n", msg, err);
    std::exit(1);
  }
}

class PS_StochTransientKernel
{
  public:
    PS_StochTransientKernel
      ( cl_device_id device_id
      , cl_context context
      , cl_double* msc_non_zero
      , cl_uint* msc_non_zero_row
      , cl_uint* msc_col_offset
      , cl_uint msc_non_zero_size
      , cl_uint msc_dim

      , cl_double* fgw_d
      );
    ~PS_StochTransientKernel();

    void run(cl_command_queue queue, cl_double* vec_in, cl_double* vec_out, cl_uint times = 1);

  private:
    cl_device_id device_id_m;
    cl_context context_m;

    cl_program program_m;
    cl_kernel kernel_m;

    cl_mem vec_out_cl_m;
    cl_mem vec_in_cl_m;
    cl_mem non_zero_cl_m;
    cl_mem non_zero_row_cl_m;
    cl_mem column_offset_cl_m;
    cl_mem fgw_d_cl_m;
    cl_uint dim_m;
    size_t non_zero_size_m;
};

#endif // PRISM_PS_STOCH_TRANSIENT_KERNEL
