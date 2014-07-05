#ifndef PRISM_OPENCL_UTIL_H
#define PRISM_OPENCL_UTIL_H

#include <CL/cl.h>

#define CLERR() { if (err != CL_SUCCESS) { std::printf("%s:%d:%d OpenCL error %d\n", __FILE__, __LINE__, __func__, err); } }
#define CLERR_PASS() { if (err != CL_SUCCESS) { return err; } }

template <typename T>
T least_greater_multiple(T a, T min)
{
  T r = a;
  while (r < min) { r += a; }
  return r;
}

struct cl_device_information
{
  cl_device_information(cl_device_id id = 0, cl_device_type type = 0) : id(id), type(type) {}
  cl_device_id id;
  cl_device_type type;
};

template <typename Out1>
static inline
int cl_get_devices
  ( Out1 devices, size_t devices_max_size, size_t& devices_size
  , cl_device_type type
  )  
{
  int err = 0;
  devices_size = 0;

  cl_platform_id pids[32];
  cl_device_id dids[32];
  cl_uint pids_size;
  cl_uint dids_size;
  
  err = clGetPlatformIDs(1, pids, &pids_size); CLERR_PASS();
  
  for (size_t ii = 0; ii < pids_size; ++ii)
  {
    err = clGetDeviceIDs(pids[ii], type, 32, dids, &dids_size); CLERR_PASS();
    for (size_t jj = 0; jj < dids_size && devices_size < devices_max_size; ++jj)
    {
      *(devices++) = cl_device_information(dids[ii], type);
      ++devices_size;
    }
  } 
}

template <typename T>
static inline
void cl_create_buffer(cl_context c, cl_mem& m, cl_mem_flags flags, size_t num, T* host_ptr = NULL)
{
  cl_int err = 0;
  m = clCreateBuffer(c, flags, num * sizeof(T), host_ptr, &err);
}

template <typename P>
static inline
void cl_fill_buffer(cl_command_queue q, cl_mem m, P pattern, size_t num, size_t offset = 0)
{
  // To be compatible with OpenCL < 1.2
  std::vector<P> tmp(num, pattern);
  cl_write_buffer(q, m, num, tmp.data());
  /*
  cl_int err = 0;
  cl_event ev;

  err = clEnqueueFillBuffer(q, m, &pattern, sizeof(P), offset, num * sizeof(P), 0, 0, &ev); 
  err = clWaitForEvents(1, &ev);
  clReleaseEvent(ev);
  */
}

template <typename T>
static inline
cl_int cl_write_buffer(cl_command_queue q, cl_mem m, size_t num, T* host_ptr, size_t offset = 0)
{
  return clEnqueueWriteBuffer(q, m, CL_TRUE, offset, num * sizeof(T), host_ptr, 0, 0, NULL); 
}


template <typename T>
cl_int cl_read_buffer(cl_command_queue q, cl_mem m, size_t num, T* host_ptr, size_t offset = 0)
{ 
  return clEnqueueReadBuffer(q, m, CL_TRUE, offset, num * sizeof(T), host_ptr, 0, 0, NULL);
}


class OpenCLKernel
{
  public:
    OpenCLKernel
      ( cl_device_id cl_device_
      , cl_context cl_context_
      , char const* source
      , char const* kernel_name
      )
      : cl_device_m(cl_device_)
      , cl_context_m(cl_context_)
      {
        cl_int err = 0;
        
        cl_queue_m = clCreateCommandQueue(cl_context_m, cl_device_m, 0, &err);
        cl_program_m = clCreateProgramWithSource(cl_context_m, 1, &source, NULL, &err);
        err = clBuildProgram(cl_program_m, 1, &cl_device_m, NULL, NULL, NULL);
        cl_kernel_m = clCreateKernel(cl_program_m, kernel_name, &err);
      }

    ~OpenCLKernel()
    {
      clReleaseKernel(cl_kernel_m);
      clReleaseProgram(cl_program_m);
      clReleaseCommandQueue(cl_queue_m);
    }

  protected:
    template <typename T>
    void cl_create_buffer(cl_mem& m, cl_mem_flags flags, size_t num, T* host_ptr = NULL)
    {
      ::cl_create_buffer<T>(cl_context_m, m, flags, num, host_ptr); 
    }
    
    template <typename P>
    void cl_fill_buffer(cl_mem m, P pattern, size_t num, size_t offset = 0)
    {
      ::cl_fill_buffer<P>(cl_queue_m, m, pattern, num, offset);
    }
    
    template <typename T>
    cl_int cl_write_buffer(cl_mem m, size_t num, T* host_ptr, size_t offset = 0)
    {
      return ::cl_write_buffer<T>(cl_queue_m, m, num, host_ptr, offset);
    }
    
    template <typename T>
    cl_int cl_read_buffer(cl_mem m, size_t num, T* host_ptr, size_t offset = 0)
    {
      return ::cl_read_buffer<T>(cl_queue_m, m, num, host_ptr, offset);
    }

    template <typename T>
    cl_int cl_set_arg(cl_uint arg_index, T arg)
    {
      return ::clSetKernelArg(cl_kernel_m, arg_index, sizeof(T), &arg);
    }
    
    cl_device_id cl_device_m;
    cl_context cl_context_m;
    cl_command_queue cl_queue_m;
    cl_program cl_program_m;
    cl_kernel cl_kernel_m;
};

#endif // PRISM_OPENCL_UTIL_H

