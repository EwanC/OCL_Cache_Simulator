__kernel void MatVecMulUncoalesced0(const __global float* M,
                                    const __global float* V,
                                    uint width, uint height,
                                    __global float* W)
{
    // Row index
    uint y = get_global_id(0);
    float dotProduct = 0;
    for (int x = 0; x < width; ++x) {
      dotProduct += M[y * width + x] * V[x];
    }
    W[y] = dotProduct;
}

// Matrix multiplication kernel called by MatrixMul()
__kernel void MatVecMulUncoalesced1(const __global float* M,
                                    const __global float* V,
                                    uint width, uint height,
                                    __global float* W)
{        
    // Each work-item handles as many matrix rows as necessary
    for (uint y = get_global_id(0);
         y < height;
         y += get_global_size(0))
    {

        // Row pointer
        const __global float* row = M + y * width;

        // Compute dot product  
        float dotProduct = 0;
        for (uint x = 0; x < width; ++x)
            dotProduct += row[x] * V[x];

        // Write result to global memory
        W[y] = dotProduct;
    }
}

// Matrix multiplication kernel called by MatrixMul()
__kernel void MatVecMulCoalesced0(const __global float* M,
                                  const __global float* V,
                                  uint width, uint height,
                                  __global float* W,
                                  __local float* partialDotProduct)
{    
    // Each work-group handles as many matrix rows as necessary
    for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {

        // Row pointer
        const __global float* row = M + y * width;
        
        // Each work-item accumulates as many products as necessary
        // into local variable "sum"
        float sum = 0;
        for (uint x = get_local_id(0); x < width; x += get_local_size(0))
            sum += row[x] * V[x];

        // Each partial dot product is stored in shared memory
        partialDotProduct[get_local_id(0)] = sum;

        // Synchronize to make sure each work-item is done updating
        // shared memory; this is necessary because in the next step,
        // the first work-item needs to read from shared memory
        // the partial dot products written by the other work-items
        barrier(CLK_LOCAL_MEM_FENCE);

        // The first work-item in the work-group adds all partial
        // dot products together and writes the result to global memory
        if (get_local_id(0) == 0) {
            float dotProduct = 0;
            for (uint t = 0; t < get_local_size(0); ++t)
                dotProduct += partialDotProduct[t];
            W[y] = dotProduct;
	      }

        // Synchronize to make sure the first work-item is done with
        // reading partialDotProduct
        barrier(CLK_LOCAL_MEM_FENCE);
	    }
}

// Matrix multiplication kernel called by MatrixMul()
__kernel void MatVecMulCoalesced1(const __global float* M,
                                  const __global float* V,
                                  uint width, uint height,
                                  __global float* W,
                                  __local float* partialDotProduct)
{    
    // Each work-group handles as many matrix rows as necessary
    for (uint y = get_group_id(0); y < height; y += get_num_groups(0)) {

        // Row pointer
        const __global float* row = M + y * width;
        
        // Each work-item accumulates as many products as necessary
        // into local variable "sum"
        float sum = 0;
        for (uint x = get_local_id(0); x < width; x += get_local_size(0))
            sum += row[x] * V[x];

        // Each partial dot product is stored in shared memory
        partialDotProduct[get_local_id(0)] = sum;
        
        // Perform parallel reduction to add each work-item's
        // partial dot product together
        for (uint stride = 1; stride < get_local_size(0); stride *= 2) {

            // Synchronize to make sure each work-item is done updating
            // shared memory; this is necessary because work-items read
            // results that have been written by other work-items
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Index into the "partialDotProduct" array where
            // the work-item will write during this step
            uint index = 2 * stride * get_local_id(0);
            
            // Check for valid indices
            if (index < get_local_size(0)) {
            
                // Add two elements from the "partialDotProduct" array
                // and store the result in partialDotProduct[index]
                partialDotProduct[index] += partialDotProduct[index + stride];
            }
        }

        // Write the result of the reduction to global memory
        if (get_local_id(0) == 0)
            W[y] = partialDotProduct[0];

        // Synchronize to make sure the first work-item is done with
        // reading partialDotProduct
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}




