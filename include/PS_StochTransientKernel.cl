#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void PS_StochTransientKernel
  ( __global double const* msc_non_zero
  , __global uint const* msc_non_zero_row
  , __global uint const* msc_column_offset
  , const uint msc_dim

  , __global double const* fgw_d
  , const double fgw_w
  , __global double* sum

  , __global double const* v0
  , __global double* v1
  )
{
  int col = get_global_id(0);
  if (col < msc_dim)
  {
    uint cb = msc_column_offset[col];
    uint ce = msc_column_offset[col + 1];

    double dot_product = fgw_d[col] * v0[col];
    for (uint i = cb; i < ce; ++i)
    {
      dot_product += msc_non_zero[i] * v0[msc_non_zero_row[i]];
    }
    v1[col] = dot_product;

    sum[col] += fgw_w * dot_product;
  }
}
