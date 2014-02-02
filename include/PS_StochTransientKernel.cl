__kernel void vector_matrix_multiplication_msc
  ( __global float const* msc_non_zero
  , __global uint const* msc_non_zero_row
  , __global uint const* msc_column_offset
  , const uint msc_column_count
  , __global float* vec_out
  , __global float const* vec_in
  )
{
	int col = get_global_id(0);
	if (col < msc_column_count)
	{
    uint cb = msc_column_offset[col];
    uint ce = msc_column_offset[col + 1];

    float dot_product = 0;
		for (uint i = cb; i < ce; ++i)
    {
      dot_product += msc_non_zero[i] * vec_in[msc_non_zero_row[i]];
    }
    vec_out[col] = dot_product;
	}
}
