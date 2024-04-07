### BlurImage_Convolution1D.cu
- RGB image blurring uses convolution between a filter and each color channel of the image.
- The program will:
    - Read the RGB input image file “in.pnm” and output image file correctly “out_target.pnm”.
    - Blur RGB input image by host (do sequentially)
    - Compare the host's result image with the correct result image to know whether the installation is correct or not.
    - Blur the RGB input image using device (do it in parallel).
    - Compare the device's result image with the correct result image to know whether the installation is correct or not.
    - Write the resulting images to file
### Reduction.cu
- Parallel execution in CUDA to optimize the "reduction" program (specifically: we will calculate the sum of an integer array).
    - reduceBlksKernel1: uses a loop with a variable stride, doubling its value each iteration until it reaches blockDim.x.
    - reduceBlksKernel2: uses a single loop over stride, doubling its value each iteration until it reaches 2 * blockDim.x.
    - reduceBlksKernel3: starts with the widest stride (blockDim.x) and halves it in each iteration until reaching 1.
- For each kernel function, the program prints:
    - Grid size and block size.
    - Kernel function run time (“kernel time”)
    - “CORRECT” if the calculated result is the same as the correct result, “INCORRECT” otherwise.
### MatrixMultiplication.cu
- Apply the knowledge learned about CUDA to install the algorithm to multiply two matrices of real numbers :
    - matrix_multiplication_kernel1: Basic Matrix Multiplication, simple installation using global memory.
    - matrix_multiplication_kernel2: Tiled Matrix Multiplication. More optimized settings, using shared memory.
- For each kernel function, the program prints:
    - Grid size and block size.
    - Runtime
    - Difference between host and device results. 
        - If the average difference has a small value (for example, 0.xxx), it is not necessarily wrong, because when calculating real numbers, there may be a small difference between the CPU and GPU.
        - If the average difference value is greater than 0.xxx then it is probably wrong.
### BlurImage_Convolution2D.cu
- Apply understanding of memory types in CUDA to optimize an RGB image blur program
    - blurImgKernel1 : looks like BlurImage_Convolution1D.cu
    - blurImgKernel2 : blurring images using SMEM
        - Each block will read its portion of data from inPixels in GMEM into SMEM
        - Then this part of data in SMEM will be reused many times for threads in the block
    - blurImgKernel3 : blurring images using SMEM for "inPixels" and uses CMEM for "filters"
- The program will:
    - Read image file in.pnm
    - Blur the image using the host (to get the correct results as standard),
    - Blur the image using device with 3 versions of the kernel function: blurImgKernel1, blurImgKernel2, and blurImgKernel3;
    - The results will be written to 4 files: out_host.pnm, out_device1.pnm,out_device2.pnm, and out_device3.pnm.
    - For each kernel function, the program prints to the screen the runtime and the difference from the host's results
### RadixSort.cu
- Parallelizing Radix Sort with Cuda Architecture
    - histogramKernel: Computes the local histogram of elements in each block, used to distribute elements into bins based on the corresponding bit.
    - scanBlkKernel: Performs a scan (prefix sum) on the local histogram to calculate the offset for the distribution of sorted elements in each block.
    - addBlkKernel: Performs accumulation on the elements in each block to calculate the final offset for the sorted elements.
    - transposeKernel: Transposes the matrix of the global histogram to prepare it for use in sorting.
    - scatterKernel: Distributes elements from the sorted in array into the out array based on information about the transposed histogram.
    - sortBlock: Sorts the elements in each block using the radix sort method.
    - scatterBlock: Distributes sorted elements from each block into the out array using histogram information.