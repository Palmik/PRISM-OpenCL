#ifndef PRISM_VECTOR_MATRIX_MULTIPLICATION_MSC
#define PRISM_VECTOR_MATRIX_MULTIPLICATION_MSC

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

class vector_matrix_multiplication_msc
{
  public:
    vector_matrix_multiplication_msc
      ( cl_device_id device_id
      , cl_context context
      , float* msc_non_zero
      , cl_uint* msc_non_zero_row
      , cl_uint* msc_col_offset
      , cl_uint msc_non_zero_size
      , cl_uint msc_dim
      );
    ~vector_matrix_multiplication_msc();

    void run(cl_command_queue queue, float* vec_in, float* vec_out, cl_uint times = 1);

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
    cl_uint dim_m;
    size_t non_zero_size_m;
};

#endif // PRISM_VECTOR_MATRIX_MULTIPLICATION_MSC
