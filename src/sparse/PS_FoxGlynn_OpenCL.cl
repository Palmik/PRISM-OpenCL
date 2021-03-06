#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef double real;

__kernel void PS_FoxGlynn
  ( const uint warp_size
  , __global real const* fw_non_zero
  , __global uint const* fw_non_zero_row
  , __global uint const* fw_seg_offset
  , const uint fw_ns
  , const uint fw_ns_rem

  , __global real const* fgw_d
  , const real fgw_w
  , __global real* sum

  , __global real const* v0
  , __global real* v1
  )
{
  int col = get_group_id(0) * get_local_size(0) + get_local_id(0);
  int seg_i = col / warp_size;
  int off_i = get_local_id(0) % warp_size;

  uint dim = (fw_ns - 1) * warp_size + fw_ns_rem;
  if (col < dim)
  {
    real dot_product = fgw_d[col] * v0[col];
    uint skip = (seg_i < fw_ns - 1) ? warp_size : fw_ns_rem;
  
    uint sb = fw_seg_offset[seg_i];
    uint se = fw_seg_offset[seg_i + 1];
    for (uint ii = sb + off_i; ii < se; ii += skip)
    {
      dot_product = fma(fw_non_zero[ii], v0[fw_non_zero_row[ii]], dot_product);
    }
    v1[col] = dot_product;

    sum[col] = fma(fgw_w, dot_product, sum[col]);
  }
}

__kernel void PS_FoxGlynn_Naive
  ( __global real const* msc_non_zero
  , __global uint const* msc_non_zero_row
  , __global uint const* msc_column_offset
  , const uint msc_dim

  , __global real const* fgw_d
  , const real fgw_w
  , __global real* sum

  , __global real const* v0
  , __global real* v1
  )
{
  int col = get_global_id(0);
  if (col < msc_dim)
  {
    uint cb = msc_column_offset[col];
    uint ce = msc_column_offset[col + 1];

    real dot_product = fgw_d[col] * v0[col];
    for (uint i = cb; i < ce; ++i)
    {
      dot_product += msc_non_zero[i] * v0[msc_non_zero_row[i]];
    }
    v1[col] = dot_product;

    sum[col] += fgw_w * dot_product;
  }
}

