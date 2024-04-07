#include <stdio.h>
#include <stdint.h>

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(1);                                               \
        }                                                          \
    }

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Sequential Radix Sort
void sortByHost(const uint32_t *in, int n,
                uint32_t *out)
{
    int *bits = (int *)malloc(n * sizeof(int));
    int *nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t *src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t *originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t *dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
    // In each loop, sort elements according to the current bit from src to dst
    // (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i - 1] + bits[i - 1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n - 1] - bits[n - 1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t *temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#define getBin(num) (((num) >> (bit)) & ((nBins)-1))

__global__ void histogramKernel(uint32_t *in, int n, int *histArr, int nBits, int bit)
{
    extern __shared__ uint32_t s_in[];
    int nBins = 1 << nBits;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = threadIdx.x; stride < nBins; stride += blockDim.x)
    {
        s_in[stride] = 0;
    }
    __syncthreads();

    // Compute local histogram
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        atomicAdd(&s_in[getBin(in[i])], 1);
    }
    __syncthreads();

    // Reduce local histograms into global histogram
    for (int stride = threadIdx.x; stride < nBins; stride += blockDim.x)
    {
        atomicAdd(&histArr[blockIdx.x + gridDim.x * stride], s_in[stride]);
    }
}

__global__ void scanBlkKernel(int *in, int n, int *out, int *blkSums)
{
    extern __shared__ int temp[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    temp[threadIdx.x] = (threadIdx.x > 0) ? in[idx - 1] : 0;
    __syncthreads();

    if (idx >= n)
    {
        return;
    }

    out[idx] = temp[threadIdx.x];
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (threadIdx.x >= stride)
        {
            out[idx] += temp[threadIdx.x - stride];
        }
        __syncthreads();
        temp[threadIdx.x] = out[idx];
        __syncthreads();
    }
    if (blkSums != NULL && (idx == n - 1 || threadIdx.x == blockDim.x - 1))
    {
        blkSums[blockIdx.x] = out[idx] + in[idx];
    }
}

__global__ void addBlkKernel(int *in, int n, int *blkSums)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x == 0 || idx >= n)
    {
        return;
    }
    in[idx] = in[idx] + blkSums[blockIdx.x - 1];
}

__global__ void transposeKernel(int *in, int n, int width, int *out)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    int x = idx % width;
    int y = idx / width;
    out[x * (n / width) + y] = in[idx];
}

__global__ void scatterKernel(uint32_t *in, int n, uint32_t *out,
                              int *scanHistogramArrayTranspose,
                              int nBits, int bit)
{
    int nBins = 1 << nBits;
    int size = blockDim.x;
    if (blockIdx.x == gridDim.x - 1)
    {
        size = n - (gridDim.x - 1) * blockDim.x;
    }
    extern __shared__ uint32_t tp[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    tp[threadIdx.x] = (idx < n) ? in[idx] : 0;

    int startHistArr = 2 * blockDim.x + 1 + nBins;
    for (int i = 0; i < nBins; i += blockDim.x)
    {
        if (threadIdx.x + i < nBins)
        {
            tp[startHistArr + threadIdx.x + i] = scanHistogramArrayTranspose[blockIdx.x * nBins + threadIdx.x + i];
        }
    }
    __syncthreads();

    int startBitArr = blockDim.x + 1;
    int startBitScan = blockDim.x;
    int nZeros = 0;

    for (int i = 0; i < nBits; i++)
    {
        if (threadIdx.x < size)
        {
            uint32_t oneBit = (getBin(tp[threadIdx.x]) >> i) & 1;
            tp[startBitArr + threadIdx.x] = oneBit;
        }

        tp[blockDim.x] = 0;
        __syncthreads();

        for (int stride = 1; stride < size; stride *= 2)
        {
            int temp = 0;
            if (threadIdx.x >= stride && threadIdx.x < size)
            {
                temp = tp[startBitScan + threadIdx.x - stride];
            }
            __syncthreads();
            if (threadIdx.x >= stride && threadIdx.x < size)
            {
                tp[startBitScan + threadIdx.x] += temp;
            }
            __syncthreads();
        }

        nZeros = size - tp[startBitScan + size - 1] - tp[startBitArr + size - 1];

        uint32_t ele;
        if (threadIdx.x < size)
        {
            ele = tp[threadIdx.x];
        }
        __syncthreads();
        if (threadIdx.x < size)
        {
            uint32_t oneBit = (getBin(ele) >> i) & 1;
            if (oneBit == 0)
            {
                int rank = threadIdx.x - tp[startBitScan + threadIdx.x];
                tp[rank] = ele;
            }
            else
            {
                int rank = nZeros + tp[startBitScan + threadIdx.x];
                tp[rank] = ele;
            }
        }
        __syncthreads();
    }

    int startArrIdx = 2 * blockDim.x + 1;
    if (threadIdx.x == 0)
    {
        int bin = getBin(tp[threadIdx.x]);
        tp[startArrIdx + bin] = 0;
    }
    else if (threadIdx.x < size)
    {
        if (getBin(tp[threadIdx.x]) != getBin(tp[threadIdx.x - 1]))
        {
            tp[startArrIdx + getBin(tp[threadIdx.x])] = threadIdx.x;
        }
    }
    __syncthreads();

    int startArrEleBef = blockDim.x;
    int bin = getBin(tp[threadIdx.x]);
    tp[startArrEleBef + threadIdx.x] = threadIdx.x - tp[startArrIdx + bin];

    if (threadIdx.x < size)
    {
        int rank = tp[startHistArr + bin] + tp[startArrEleBef + threadIdx.x];
        out[rank] = tp[threadIdx.x];
    }
}

__global__ void sortBlock(uint32_t *in, int n, int nBits, int bit)
{
    int nBins = 1 << nBits;
    int size = blockDim.x;
    if (blockIdx.x == gridDim.x - 1)
    {
        size = n - (gridDim.x - 1) * blockDim.x;
    }

    extern __shared__ uint32_t tp[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    tp[threadIdx.x] = (idx < n) ? in[idx] : 0;
    __syncthreads();

    int startBitArr = blockDim.x + 1;
    int startBitScan = blockDim.x;
    int nZeros = 0;

    for (int i = 0; i < nBits; i++)
    {

        if (threadIdx.x < size)
        {
            uint32_t oneBit = (getBin(tp[threadIdx.x]) >> i) & 1;
            tp[startBitArr + threadIdx.x] = oneBit;
        }

        tp[blockDim.x] = 0;
        __syncthreads();

        for (int stride = 1; stride < size; stride *= 2)
        {
            int temp = 0;
            if (threadIdx.x >= stride && threadIdx.x < size)
            {
                temp = tp[startBitScan + threadIdx.x - stride];
            }
            __syncthreads();
            if (threadIdx.x >= stride && threadIdx.x < size)
            {
                tp[startBitScan + threadIdx.x] += temp;
            }
            __syncthreads();
        }

        nZeros = size - tp[startBitScan + size - 1] - tp[startBitArr + size - 1];

        uint32_t ele;
        if (threadIdx.x < size)
        {
            ele = tp[threadIdx.x];
        }
        __syncthreads();
        if (threadIdx.x < size)
        {
            uint32_t oneBit = (getBin(ele) >> i) & 1;
            if (oneBit == 0)
            {
                int rank = threadIdx.x - tp[startBitScan + threadIdx.x];
                tp[rank] = ele;
            }
            else
            {
                int rank = nZeros + tp[startBitScan + threadIdx.x];
                tp[rank] = ele;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < size)
    {
        in[idx] = tp[threadIdx.x];
    }
}

__global__ void scatterBlock(uint32_t *in, int n, uint32_t *out, int *scanHistogramArrayTranspose, int nBits, int bit)
{
    int nBins = 1 << nBits;
    int size = blockDim.x;
    if (blockIdx.x == gridDim.x - 1)
    {
        size = n - (gridDim.x - 1) * blockDim.x;
    }

    extern __shared__ uint32_t tp[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    tp[threadIdx.x] = (idx < n) ? in[idx] : 0;
    int startHistArr = 2 * blockDim.x + nBins;

    for (int i = 0; i < nBins; i += blockDim.x)
    {
        if (threadIdx.x + i < nBins)
        {
            tp[startHistArr + threadIdx.x + i] = scanHistogramArrayTranspose[blockIdx.x * nBins + threadIdx.x + i];
        }
    }
    __syncthreads();

    int startArrIdx = 2 * blockDim.x;
    if (threadIdx.x == 0)
    {
        int bin = getBin(tp[threadIdx.x]);
        tp[startArrIdx + bin] = 0;
    }
    else if (threadIdx.x < size)
    {
        if (getBin(tp[threadIdx.x]) != getBin(tp[threadIdx.x - 1]))
        {
            tp[startArrIdx + getBin(tp[threadIdx.x])] = threadIdx.x;
        }
    }
    __syncthreads();

    int startArrEleBef = blockDim.x;
    int bin = getBin(tp[threadIdx.x]);
    tp[startArrEleBef + threadIdx.x] = threadIdx.x - tp[startArrIdx + bin];

    if (threadIdx.x < size)
    {
        int rank = tp[startHistArr + bin] + tp[startArrEleBef + threadIdx.x];
        out[rank] = tp[threadIdx.x];
    }
}

void sortByDevice(const uint32_t *in, int n, uint32_t *out, int blockSizes)
{
    int nBits = 8;
    int nBins = 1 << nBits;
    int gridSizeHist = (n - 1) / blockSizes + 1;
    int gridSizeScan = (gridSizeHist * nBins - 1) / blockSizes + 1;

    int in_size = n * sizeof(uint32_t);
    int out_size = in_size;
    uint32_t *d_src, *d_dst;
    CHECK(cudaMalloc(&d_src, in_size));
    CHECK(cudaMalloc(&d_dst, out_size));
    CHECK(cudaMemcpy(d_src, in, in_size, cudaMemcpyHostToDevice));

    size_t histArr_size = gridSizeHist * nBins * sizeof(int);
    size_t size_blksum = gridSizeScan * sizeof(int);
    int *blkSums = (int *)malloc(size_blksum);
    int *d_histArr, *d_scanHistArr, *d_blkSums;
    CHECK(cudaMalloc(&d_histArr, histArr_size));
    CHECK(cudaMalloc(&d_scanHistArr, histArr_size));
    CHECK(cudaMalloc(&d_blkSums, size_blksum));

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        CHECK(cudaMemset(d_histArr, 0, histArr_size));

        histogramKernel<<<gridSizeHist, blockSizes, (nBins) * sizeof(uint32_t)>>>(d_src, n, d_histArr, nBits, bit);
        CHECK(cudaGetLastError());

        scanBlkKernel<<<gridSizeScan, blockSizes, blockSizes * sizeof(int)>>>(d_histArr, gridSizeHist * nBins, d_scanHistArr, d_blkSums);
        CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(blkSums, d_blkSums, size_blksum, cudaMemcpyDeviceToHost));

        for (int i = 1; i < gridSizeScan; ++i)
        {
            blkSums[i] += blkSums[i - 1];
        }

        CHECK(cudaMemcpy(d_blkSums, blkSums, size_blksum, cudaMemcpyHostToDevice));

        addBlkKernel<<<gridSizeScan, blockSizes>>>(d_scanHistArr, gridSizeHist * nBins, d_blkSums);
        CHECK(cudaGetLastError());

        transposeKernel<<<gridSizeScan, blockSizes>>>(d_scanHistArr, gridSizeHist * nBins, gridSizeHist, d_histArr);
        CHECK(cudaGetLastError());

        sortBlock<<<gridSizeHist, blockSizes, (2 * blockSizes + 1) * sizeof(uint32_t)>>>(d_src, n, nBits, bit);
        CHECK(cudaGetLastError());
        scatterBlock<<<gridSizeHist, blockSizes, (2 * blockSizes + 2 * nBins) * sizeof(uint32_t)>>>(d_src, n, d_dst, d_histArr, nBits, bit);
        CHECK(cudaGetLastError());

        uint32_t *tp = d_src;
        d_src = d_dst;
        d_dst = tp;
    }

    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_histArr));
    CHECK(cudaFree(d_scanHistArr));
    CHECK(cudaFree(d_blkSums))
    free(blkSums);
}

// Radix Sort
void sort(const uint32_t *in, int n,
          uint32_t *out,
          bool useDevice = false, int blockSize = 1)
{
    GpuTimer timer;
    timer.Start();

    if (useDevice == false)
    {
        printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
        printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t *out, uint32_t *correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t *a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char **argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    // int n = 50; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t *in = (uint32_t *)malloc(bytes);
    uint32_t *out = (uint32_t *)malloc(bytes);        // Device result
    uint32_t *correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        // in[i] = rand() % 255; // For test by eye
        in[i] = rand();
    }
    // printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    // printArray(correctOut, n); // For test by eye

    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    // printArray(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);

    return EXIT_SUCCESS;
}