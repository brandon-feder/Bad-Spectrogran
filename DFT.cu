
// Computes the DFT for coefficient k
__device__
unsigned int X(unsigned int *x, int k, int N)
{
    float sum1 = 0;

    for(int n = 0; n < N; n++)
    {
        sum1 += (float)x[n]*cos( (float)(2*k*n) * M_PI/ (float)N );
    }

    float sum2 = 0;

    for(int n = 0; n < N; n++)
    {
        sum2 += (float)x[n]*sin( (float)(2*k*n) * M_PI/ (float)N );
    }

    return sum1*sum1/10000 + sum2*sum2/10000;
}


__global__
void DFT(unsigned int *frame, unsigned int *dft)
{
    // Get the index of the current core
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // If the core is in range to be used
    if( index < FRAME_SIZE/2)
    {
        // Get the coefficient coressponding to teh cores index
        dft[ index ] = X( frame, index, FRAME_SIZE/2 );
    }

}
