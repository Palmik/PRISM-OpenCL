#ifndef PRISM_PS_FOX_GLYNN_OPENCL
#define PRISM_PS_FOX_GLYNN_OPENCL

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include <vector>
#include "jnipointer.h"

typedef cl_double cl_real;

// This represents MSR/MSC (CSR/CSC) matrix.
struct MatrixCS
{
  cl_real* non_zero;
  cl_uint* non_zero_xs; // non_zero_xs[i] is the row (col) of the value at non_zero[i]
  cl_uint* non_zero_ys_offset; // non_zero_ys_offset[i] is the offset of the i-th col (row) in the non_zero array
  cl_uint non_zero_cnt;
  cl_uint n; // the number of cols (rows)
};


template <typename FGCore>
void FGOpenCL_
  ( JNIEnv* env
   
  , MatrixCS matrix
  , cl_real* fgw_ds // fgw diagonals
  , cl_real* fgw_ws // fgw weights
  , cl_uint fgw_l 
  , cl_uint fgw_r

  , cl_real* soln1 // in
  , cl_real* soln2 // out
  , cl_real* sum
 
  , long int time
  , cl_real unif
  
  , long int& num_iters

  , long int start2
  , long int start3
  
  , bool is_cumul_reward 
  );

void FGOpenCL
  ( JNIEnv* env
   
  , MatrixCS matrix
  , cl_real* fgw_ds // fgw diagonals
  , cl_real* fgw_ws // fgw weights
  , cl_uint fgw_l 
  , cl_uint fgw_r

  , cl_real* soln1 // in
  , cl_real* soln2 // out
  , cl_real* sum
 
  , long int time
  , cl_real unif
  
  , long int& num_iters

  , long int start2
  , long int start3
  
  , bool is_cumul_reward 
  );


#endif // PRISM_PS_FOX_GLYNN_OPENCL
