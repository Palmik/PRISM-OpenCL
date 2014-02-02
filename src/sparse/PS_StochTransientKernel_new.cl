__kernel PC_StochTransientKernel
  ( __global float const* msc_non_zero     // change: setup
  , __global uint const* msc_non_zero_row  // change: setup
  , __global uint const* msc_column_offset // change: setup
  , const uint dim // change: setup
  , __gloabl float const* fgw_w // change: setup
  , __global float const* fgw_d // change: setup
  , const uint fgw_left // change: setup

  , __global uint* iteration // change: setup (and inside kernel)
  , __global float const* vec_i // change: iteration (clEnqeueCopyBuffer)
  , __global float*       vec_o // change: iteration (clEnqeueCopyBuffer)
  , __global float*       sum // change: setup (and inside kernel) 
  )
{
	int col = get_global_id(0);
	if (col < dim)
	{
    float dot_product = fgw_d[col] * vec_i[col];
    
    // MSC multiplication.
    uint cb = msc_column_offset[col];
    uint ce = msc_column_offset[col + 1];

		for (uint i = cb; i < ce; ++i)
    {
      dot_product += msc_non_zero[i] * vec_i[msc_non_zero_row[i]];
    }
    vec_o[col] = dot_product;

    if (iteration[0] >= fgw_left)
    {
      sum[col] += fgw_w[iteration[0] - fgw_left] * dot_product;
    }

    if (col + 1 == dim)
    {
      iteration[0] += 1;
    }
  }
}
