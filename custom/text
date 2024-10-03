
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_WIDTH 16

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
//     //(void)Width_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int W_grid = ceil((float)Width_out / TILE_WIDTH);
//     int m = blockIdx.x;
//     int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
//     int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
//     int z = blockIdx.z;
//     float acc = 0.0f;


//      if(h < Height_out && w < Width_out){
//     for(int c = 0; c < Channel; c++){
//         for(int p = 0; p < K; p++){
//             for(int q = 0; q < K; q++){
//                 acc += in_4d(z, c, h + p, w + q) * mask_4d(m, c, p, q);
               
//             }
//         }
//     }
    
//         out_4d(z, m, h, w) =  acc;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

// __global__ void tiled_conv(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, int W_grid){
//     int h0, w0, h_base, w_base, h, w;
//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     int X_tile_width = TILE_WIDTH + K - 1;
//     //int Y_tile_width = Height;
//     extern __shared__ float shmem[];
//     float * X_shared = &shmem[0];
//     float * W_shared = &shmem[X_tile_width * X_tile_width];
//     int n = blockIdx.z;
//     int m = blockIdx.x; //current mask set we are on 
//     h0 = threadIdx.x;
//     w0 = threadIdx.y;
//     h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
//     w_base = (blockIdx.y % W_grid) * TILE_WIDTH;
//     h = h_base + h0;
//     w = w_base + w0;

//     float acc = 0.0;

//     for(int c = 0; c < Channel; c++){
//         if ((h0 < K) && (w0 < K))
//         W_shared[h0 * K + w0] = mask_4d(m,c,h0, w0);
//         __syncthreads();

//         for(int i = h; i < h_base + X_tile_width; i+= TILE_WIDTH){
//             for(int j = w; j < w_base + X_tile_width; j+= TILE_WIDTH)
        
//             X_shared[(X_tile_width)*(i - h_base) + j - w_base] = in_4d(n,c,h,w);
//         }
//         __syncthreads();

//         for(int p = 0; p < K; p++){
//             for(int q = 0; q < K; q++)
//             acc = acc + X_shared[(X_tile_width) * (h+p) + w + q] * W_shared[p * K + q];
//         }
//         __syncthreads();
//     }

//     if(h < Height_out && w < Width_out)
//     out_4d(n,m,h,w) = acc;

//     }


	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Allocate memory and copy over the relevant data structures to the GPU

//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }

    
    
//     cudaStream_t cudaArr[10];
//     for(int i = 0; i < 10; i++){
//         cudaStreamCreate(&cudaArr[i]);
//     }
//     int seg_size = 10;
    
//     cudaMalloc((void**) device_output_ptr, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1));
//     cudaMalloc((void**) device_input_ptr, sizeof(float) * Batch * Channel * Height * Width);
//     cudaMalloc((void**) device_mask_ptr, sizeof(float) * Channel * Map_out * K * K);

    
   
    
//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;

//     int W_grid = ceil((float)W_out / TILE_WIDTH);
//     int H_grid = ceil((float)H_out/TILE_WIDTH);
//     int Y = W_grid * H_grid;

//     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//     dim3 dimGrid(Map_out, Y, seg_size);
//     //dim3 dimGrid(Map_out, Y, Batch);

//     int inter_batch = Channel * Height * Width;
//     int inter_out = Map_out * (Height - K + 1) * (Width - K + 1);
//     float * temp = (float*) *device_output_ptr;
//     float * temp2 = (float*) host_output;
    
//     cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Channel * Map_out * K * K, cudaMemcpyHostToDevice);

    
//     cudaHostRegister(&temp2, Batch * inter_out, cudaHostRegisterDefault);
//     cudaHostRegister(&host_input, Batch * inter_batch, cudaHostRegisterDefault);
//     //size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
    
//    for(int i = 0; i < Batch; i+= seg_size * 10){
    
//     for(int s = 0; s < 10; s++){
//         cudaMemcpyAsync(*device_input_ptr + (i + s * seg_size) * inter_batch, host_input + (i + s * seg_size) * inter_batch, seg_size * sizeof(float) * inter_batch, cudaMemcpyHostToDevice, cudaArr[s]);
//     }


//     for(int s = 0; s < 10; s++){
//         conv_forward_kernel<<<dimGrid, dimBlock, 0, cudaArr[s]>>>(*device_output_ptr + (i + s * seg_size) * inter_out,*device_input_ptr + (i + s * seg_size) * inter_batch, *device_mask_ptr, seg_size, Map_out, Channel, Height, Width, K);
//     }
    

//     for(int s = 0; s < 10; s++){
//         cudaMemcpyAsync(temp2 + (i + s * seg_size) * inter_out, temp + (i + s * seg_size) * inter_out, seg_size * inter_out * sizeof(float), cudaMemcpyDeviceToHost, cudaArr[s]);
//     }

   
//    }
   
   
//     //cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * Batch * Channel * Height * Width, cudaMemcpyHostToDevice);
//     //cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Channel * Map_out * K * K, cudaMemcpyHostToDevice);

//     //conv_forward_kernel<<<dimGrid, dimBlock>>>(*device_output_ptr, *device_input_ptr, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
//     //tiled_conv<<<dimGrid, dimBlock, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K, W_grid);
    
//     //cudaMemcpy((void *) temp2, (void*) temp, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaMemcpyDeviceToHost);

//     cudaDeviceSynchronize();

    
//     for(int i = 0; i < 10; i++){
//         cudaStreamDestroy(cudaArr[i]);
//    }
    




   
//     // Free device memory
//     cudaFree(*device_output_ptr);
//     cudaFree(*device_input_ptr);
//     cudaFree(*device_mask_ptr);
    

   

// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
//     /*
//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;

//     int W_grid = ceil((float)W_out / TILE_WIDTH);
//     int H_grid = ceil((float)H_out/TILE_WIDTH);
//     int Y = W_grid * H_grid;

//     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, Channel);
//     dim3 dimGrid(Map_out, Y, Batch);

//     //size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);

//     conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//     //tiled_conv<<<dimGrid, dimBlock, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K, W_grid);
//     */
    
//    return;
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {

//     /*
//     // Copy the output back to host
//     cudaMemcpy(host_output, device_output, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_output);
//     cudaFree(device_input);
//     cudaFree(device_mask);
    
//    */
//      return;

// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }


//CONSTANT MEMORY OPTIMIZATION
__constant__ float MASK[4000];

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) MASK[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil((float)Width_out / TILE_WIDTH);
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int z = blockIdx.z;
    float acc = 0.0f;


     if(h < Height_out && w < Width_out){
    for(int c = 0; c < Channel; c++){
        for(int p = 0; p < K; p++){
            for(int q = 0; q < K; q++){
                acc += in_4d(z, c, h + p, w + q) * mask_4d(m, c, p, q);
               
            }
        }
    }
    
        out_4d(z,m,h,w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void tiled_conv(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, int W_grid){
    int h0, w0, h_base, w_base, h, w;
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int X_tile_width = TILE_WIDTH + K - 1;
    //int Y_tile_width = Height;
    extern __shared__ float shmem[];
    float * X_shared = &shmem[0];
    float * W_shared = &shmem[X_tile_width * X_tile_width];
    int n = blockIdx.z;
    int m = blockIdx.x; //current mask set we are on 
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.y % W_grid) * TILE_WIDTH;
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0.0;

    for(int c = 0; c < Channel; c++){
        if ((h0 < K) && (w0 < K))
        W_shared[h0 * K + w0] = mask_4d(m,c,h0, w0);
        __syncthreads();

        for(int i = h; i < h_base + X_tile_width; i+= TILE_WIDTH){
            for(int j = w; j < w_base + X_tile_width; j+= TILE_WIDTH)
        
            X_shared[(X_tile_width)*(i - h_base) + j - w_base] = in_4d(n,c,h,w);
        }
        __syncthreads();

        for(int p = 0; p < K; p++){
            for(int q = 0; q < K; q++)
            acc = acc + X_shared[(X_tile_width) * (h+p) + w + q] * W_shared[p * K + q];
        }
        __syncthreads();
    }

    if(h < Height_out && w < Width_out)
    out_4d(n,m,h,w) = acc;

    }


	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    
    /*
    cudaStream_t cudaArr[10];
    for(int i = 0; i < 10; i++){
        cudaStreamCreate(&cudaArr[i]);
    }
    int seg_size = 10;
    */
    cudaMalloc((void**) device_output_ptr, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1));
    cudaMalloc((void**) device_input_ptr, sizeof(float) * Batch * Channel * Height * Width);
    //cudaMalloc((void**) device_mask_ptr, sizeof(float) * Channel * Map_out * K * K);

    
   
    /*
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;

    int W_grid = ceil((float)W_out / TILE_WIDTH);
    int H_grid = ceil((float)H_out/TILE_WIDTH);
    int Y = W_grid * H_grid;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, Channel);
    //dim3 dimGrid(Map_out, Y, seg_size);
    dim3 dimGrid(Map_out, Y, Batch);

    int inter_batch = Channel * Height * Width;
    int inter_out = Map_out * (Height - K + 1) * (Width - K + 1);
    float * temp = (float*) *device_output_ptr;
    float * temp2 = (float*) host_output;
    */
    //cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Channel * Map_out * K * K, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(MASK, host_mask,  Channel * Map_out * K * K* sizeof(float));
    
    //cudaHostRegister(&temp2, Batch * inter_out, cudaHostRegisterDefault);
    //cudaHostRegister(&host_input, Batch * inter_batch, cudaHostRegisterDefault);
    //size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
    /*
   for(int i = 0; i < Batch; i+= seg_size * 10){
    
    for(int s = 0; s < 10; s++){
        cudaMemcpyAsync(*device_input_ptr + (i + s * seg_size) * inter_batch, host_input + (i + s * seg_size) * inter_batch, seg_size * sizeof(float) * inter_batch, cudaMemcpyHostToDevice, cudaArr[s]);
    }


    for(int s = 0; s < 10; s++){
        conv_forward_kernel<<<dimGrid, dimBlock, 0, cudaArr[s]>>>(*device_output_ptr + (i + s * seg_size) * inter_out,*device_input_ptr + (i + s * seg_size) * inter_batch, *device_mask_ptr, seg_size, Map_out, Channel, Height, Width, K);
    }
    

    for(int s = 0; s < 10; s++){
        cudaMemcpyAsync(temp2 + (i + s * seg_size) * inter_out, temp + (i + s * seg_size) * inter_out, seg_size * inter_out * sizeof(float), cudaMemcpyDeviceToHost, cudaArr[s]);
    }

   
   }
   */
   
    cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * Batch * Channel * Height * Width, cudaMemcpyHostToDevice);
    //cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Channel * Map_out * K * K, cudaMemcpyHostToDevice);

    //conv_forward_kernel<<<dimGrid, dimBlock>>>(*device_output_ptr, *device_input_ptr, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //tiled_conv<<<dimGrid, dimBlock, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K, W_grid);
    
    //cudaMemcpy((void *) temp2, (void*) temp, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaMemcpyDeviceToHost);

    //cudaDeviceSynchronize();

    
    //for(int i = 0; i < 10; i++){
    //    cudaStreamDestroy(cudaArr[i]);
   // }
    




   
    // Free device memory
    //cudaFree(*device_output_ptr);
    //cudaFree(*device_input_ptr);
    //cudaFree(*device_mask_ptr);
    

   

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;

    int W_grid = ceil((float)W_out / TILE_WIDTH);
    int H_grid = ceil((float)H_out/TILE_WIDTH);
    int Y = W_grid * H_grid;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(Map_out, Y, Batch);

    //size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    //tiled_conv<<<dimGrid, dimBlock, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K, W_grid);
    
    
   return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
    
   return;
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}






// //TILED CONVOLUTION APPROACH
// __constant__ float MASK[4000];

// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_WIDTH 16

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
//     //(void)Width_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) MASK[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int W_grid = ceil((float)Width_out / TILE_WIDTH);
//     int m = blockIdx.x;
//     int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
//     int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
//     int z = blockIdx.z;
//     float acc = 0.0f;


//      if(h < Height_out && w < Width_out){
//     for(int c = 0; c < Channel; c++){
//         for(int p = 0; p < K; p++){
//             for(int q = 0; q < K; q++){
//                 acc += in_4d(z, c, h + p, w + q) * mask_4d(m, c, p, q);
               
//             }
//         }
//     }
    
//         out_4d(z,m,h,w) = acc;
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

// __global__ void tiled_conv(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, int W_grid){
//     int h0, w0, h_base, w_base, h, w;
//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     int X_tile_width = TILE_WIDTH + K - 1;
//     //int Y_tile_width = Height;
//     extern __shared__ float shmem[];
//     float * X_shared = &shmem[0];
//     float * W_shared = &shmem[X_tile_width * X_tile_width];
//     int n = blockIdx.z;
//     int m = blockIdx.x; //current mask set we are on 
//     h0 = threadIdx.x;
//     w0 = threadIdx.y;
//     h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
//     w_base = (blockIdx.y % W_grid) * TILE_WIDTH;
//     h = h_base + h0;
//     w = w_base + w0;

//     float acc = 0.0;

//     for(int c = 0; c < Channel; c++){
//         if ((h0 < K) && (w0 < K))
//         W_shared[h0 * K + w0] = mask_4d(m,c,h0, w0);
//         __syncthreads();

//         for(int i = h; i < h_base + X_tile_width; i+= TILE_WIDTH){
//             for(int j = w; j < w_base + X_tile_width; j+= TILE_WIDTH)
        
//             X_shared[(X_tile_width)*(i - h_base) + j - w_base] = in_4d(n,c,h,w);
//         }
//         __syncthreads();

//         for(int p = 0; p < K; p++){
//             for(int q = 0; q < K; q++)
//             acc = acc + X_shared[(X_tile_width) * (h+p) + w + q] * W_shared[p * K + q];
//         }
//         __syncthreads();
//     }

//     if(h < Height_out && w < Width_out)
//     out_4d(n,m,h,w) = acc;

//     }


	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Allocate memory and copy over the relevant data structures to the GPU

//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }

    
//     /*
//     cudaStream_t cudaArr[10];
//     for(int i = 0; i < 10; i++){
//         cudaStreamCreate(&cudaArr[i]);
//     }
//     int seg_size = 10;
//     */
//     cudaMalloc((void**) device_output_ptr, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1));
//     cudaMalloc((void**) device_input_ptr, sizeof(float) * Batch * Channel * Height * Width);
//     //cudaMalloc((void**) device_mask_ptr, sizeof(float) * Channel * Map_out * K * K);

    
   
//     /*
//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;

//     int W_grid = ceil((float)W_out / TILE_WIDTH);
//     int H_grid = ceil((float)H_out/TILE_WIDTH);
//     int Y = W_grid * H_grid;

//     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, Channel);
//     //dim3 dimGrid(Map_out, Y, seg_size);
//     dim3 dimGrid(Map_out, Y, Batch);

//     int inter_batch = Channel * Height * Width;
//     int inter_out = Map_out * (Height - K + 1) * (Width - K + 1);
//     float * temp = (float*) *device_output_ptr;
//     float * temp2 = (float*) host_output;
//     */
//     //cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Channel * Map_out * K * K, cudaMemcpyHostToDevice);
//     cudaMemcpyToSymbol(MASK, host_mask,  Channel * Map_out * K * K* sizeof(float));
    
//     //cudaHostRegister(&temp2, Batch * inter_out, cudaHostRegisterDefault);
//     //cudaHostRegister(&host_input, Batch * inter_batch, cudaHostRegisterDefault);
//     //size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
//     /*
//    for(int i = 0; i < Batch; i+= seg_size * 10){
    
//     for(int s = 0; s < 10; s++){
//         cudaMemcpyAsync(*device_input_ptr + (i + s * seg_size) * inter_batch, host_input + (i + s * seg_size) * inter_batch, seg_size * sizeof(float) * inter_batch, cudaMemcpyHostToDevice, cudaArr[s]);
//     }


//     for(int s = 0; s < 10; s++){
//         conv_forward_kernel<<<dimGrid, dimBlock, 0, cudaArr[s]>>>(*device_output_ptr + (i + s * seg_size) * inter_out,*device_input_ptr + (i + s * seg_size) * inter_batch, *device_mask_ptr, seg_size, Map_out, Channel, Height, Width, K);
//     }
    

//     for(int s = 0; s < 10; s++){
//         cudaMemcpyAsync(temp2 + (i + s * seg_size) * inter_out, temp + (i + s * seg_size) * inter_out, seg_size * inter_out * sizeof(float), cudaMemcpyDeviceToHost, cudaArr[s]);
//     }

   
//    }
//    */
   
//     cudaMemcpy(*device_input_ptr, host_input, sizeof(float) * Batch * Channel * Height * Width, cudaMemcpyHostToDevice);
//     //cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Channel * Map_out * K * K, cudaMemcpyHostToDevice);

//     //conv_forward_kernel<<<dimGrid, dimBlock>>>(*device_output_ptr, *device_input_ptr, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
//     //tiled_conv<<<dimGrid, dimBlock, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K, W_grid);
    
//     //cudaMemcpy((void *) temp2, (void*) temp, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaMemcpyDeviceToHost);

//     //cudaDeviceSynchronize();

    
//     //for(int i = 0; i < 10; i++){
//     //    cudaStreamDestroy(cudaArr[i]);
//    // }
    




   
//     // Free device memory
//     //cudaFree(*device_output_ptr);
//     //cudaFree(*device_input_ptr);
//     //cudaFree(*device_mask_ptr);
    

   

// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Set the kernel dimensions and call the kernel
    
//     int W_out = Width - K + 1;
//     int H_out = Height - K + 1;

//     int W_grid = ceil((float)W_out / TILE_WIDTH);
//     int H_grid = ceil((float)H_out/TILE_WIDTH);
//     int Y = W_grid * H_grid;

//     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//     dim3 dimGrid(Map_out, Y, Batch);

//     //size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);

//     conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//     //tiled_conv<<<dimGrid, dimBlock, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K, W_grid);
    
    
//    return;
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {

    
//     // Copy the output back to host
//     cudaMemcpy(host_output, device_output, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_output);
//     cudaFree(device_input);
//     cudaFree(device_mask);
    
//    return;
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }


