
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

//Sweeping various parameters to find best values (block sizes, amount of thread coarsening) (0.5 point)
#define TILE_WIDTH 20
//#define TILE_WIDTH 16

//Weight matrix (kernel values) in constant memory (0.5 point)
__constant__ float constantmask[8000];
//     #define mask_4d(i3, i2, i1, i0) constantmask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//    cudaMemcpyToSymbol(constantmask, host_mask, M*C*K*K*sizeof(float));

//Tuning, constant memory, sweeping - FINAL SUBMISSION OPTIMIZATION
__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) constantmask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int temp = (ceil(W_out/(1.0*TILE_WIDTH)));

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int h = (blockIdx.z/temp)*TILE_WIDTH+threadIdx.y;
    int w = (blockIdx.z%temp)*TILE_WIDTH+threadIdx.x;

    if (w<W_out) 
    {
        if(h<H_out)
        {
            if(by<M)
            {
                float f = 0.0f;
                #pragma unroll(8)
                for (int channel=0; channel<C; channel++) 
                {
                    #pragma unroll(6)
                    for (int p = 0; p < K; p++) 
                    {
                        #pragma unroll(6)
                        for (int q = 0; q < K; q++) 
                        {
                            f+=in_4d(bx, channel, h*S+p, w*S+q)*mask_4d(by,channel,p,q);
                        }
                    }
                }
                out_4d(blockIdx.x, by, h, w) = f;
                //atomicAdd(&out_4d(blockIdx.x, by, h, w), f);


            }
        }
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

//Tuning with restrict and loop unrolling (considered as one optimization only if you do both), sweeping, constant
// __global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     S - stride step length
//     */

//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;
//     (void)H_out; // silence declared but never referenced warning. remove this line when you start working
//     (void)W_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define mask_4d(i3, i2, i1, i0) constantmask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int temp = (ceil(W_out/(1.0*TILE_WIDTH)));

//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     int h = (blockIdx.z/temp)*TILE_WIDTH+threadIdx.y;
//     int w = (blockIdx.z%temp)*TILE_WIDTH+threadIdx.x;

//     if (w<W_out) 
//     {
//         if(h<H_out)
//         {
//             if(by<M)
//             {
//                 float f = 0.0f;
//                 #pragma unroll(8)
//                 for (int channel=0; channel<C; channel++) 
//                 {
//                     #pragma unroll(6)
//                     for (int p = 0; p < K; p++) 
//                     {
//                         #pragma unroll(6)
//                         for (int q = 0; q < K; q++) 
//                         {
//                             f+=in_4d(bx, channel, h*S+p, w*S+q)*mask_4d(by,channel,p,q);
//                         }
//                     }
//                 }
//                 out_4d(blockIdx.x, by, h, w) = f;

//             }
//         }
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }


// //FP16 arithmetic. (note this can modify model accuracy slightly), sweeping, constant
// #include <cuda_fp16.h>

// __global__ void conv_forward_kernel(float * output, const float * input, const float *  mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     S - stride step length
//     */

//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;
//     (void)H_out; // silence declared but never referenced warning. remove this line when you start working
//     (void)W_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define mask_4d(i3, i2, i1, i0) constantmask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//      int temp = (ceil(W_out / (1.0 * TILE_WIDTH)));
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int h = (blockIdx.z / temp) * TILE_WIDTH + threadIdx.y;
//     int w = (blockIdx.z % temp) * TILE_WIDTH + threadIdx.x;

//     if (w < W_out) 
//     {
//         if (h < H_out) 
//         {
//             if (by < M) 
//             {
//                 float temp_sum = 0.0f;
//                 for (int channel = 0; channel < C; channel++) 
//                 {
//                     for (int p = 0; p < K; p++)
//                      {
//                         for (int q = 0; q < K; q++) 
//                         {
//                             __half sum = __float2half(0.0f);
//                             __half v1 = __float2half(in_4d(bx, channel, h * S + p, w * S + q));
//                             __half v2 = __float2half(mask_4d(by, channel, p, q));
//                             __half multval = __hmul(v1, v2);
//                             sum = __hadd(sum, multval);
//                             temp_sum += __half2float(sum);
//                         }
//                     }
//                 }
//                 out_4d(bx, by, h, w) = temp_sum;
//                 //atomicAdd(&out_4d(blockIdx.x, by, h, w), temp_sum);

//             }
//         }
//     }
// }


// //Input channel reduction: atomics, sweeping, constant memory
// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     S - stride step length
//     */

//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;
//     (void)H_out; // silence declared but never referenced warning. remove this line when you start working
//     (void)W_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define mask_4d(i3, i2, i1, i0) constantmask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int temp = (ceil(W_out/(1.0*TILE_WIDTH)));

//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     int h = (blockIdx.z/temp)*TILE_WIDTH+threadIdx.y;
//     int w = (blockIdx.z%temp)*TILE_WIDTH+threadIdx.x;

//     if (w<W_out) 
//     {
//         if(h<H_out)
//         {
//             if(by<M)
//             {
//                 float f = 0.0f;
//                 //for (int channel=0; channel<C; channel++) 
//                 //{
//                     for (int p = 0; p < K; p++) 
//                     {
//                         for (int q = 0; q < K; q++) 
//                         {
//                             f+=in_4d(bx, blockIdx.x, h*S+p, w*S+q)*mask_4d(by,blockIdx.x,p,q);
//                         }
//                     }
//                 //}
//                 //ATOMIC ADD OPTIMIZATION
//                 atomicAdd(&out_4d(blockIdx.x, by, h, w), f);
//                 //out_4d(blockIdx.x, by, h, w) = f;

//             }
//         }
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

// //BASELINE
// __global__ void conv_forward_kernel(float * output, const float *  input, const float *  mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     S - stride step length
//     */

//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;
//     (void)H_out; // silence declared but never referenced warning. remove this line when you start working
//     (void)W_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int temp = (ceil(W_out/(1.0*TILE_WIDTH)));

//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     int h = (blockIdx.z/temp)*TILE_WIDTH+threadIdx.y;
//     int w = (blockIdx.z%temp)*TILE_WIDTH+threadIdx.x;

//     if (w<W_out) 
//     {
//         if(h<H_out)
//         {
//             if(by<M)
//             {
//                 float f = 0.0f;
//                 for (int channel=0; channel<C; channel++) 
//                 {
//                     for (int p = 0; p < K; p++) 
//                     {
//                         for (int q = 0; q < K; q++) 
//                         {
//                             f+=in_4d(bx, channel, h*S+p, w*S+q)*mask_4d(by,channel,p,q);
//                         }
//                     }
//                 }
//                 out_4d(blockIdx.x, by, h, w) = f;

//             }
//         }
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
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

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    cudaMalloc((void **)device_output_ptr, B*M*H_out*W_out*sizeof(float));
    cudaMalloc((void **)device_input_ptr, B*C*H*W*sizeof(float));
    //cudaMalloc((void **)device_mask_ptr, M*C*K*K*sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(*device_mask_ptr, host_mask, M*C*K*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constantmask, host_mask, M*C*K*K*sizeof(float));
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); 
    dim3 gridDim(B,M,(ceil(W_out / (float)TILE_WIDTH) * ceil(H_out / (float)TILE_WIDTH)));
    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMemcpy(host_output, device_output, B*M*H_out*W_out*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

}


// //STREAMS OPTIMIZATION
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"
// #define TILE_WIDTH 16


// __global__ void conv_forward_kernel(float * output, const float *  input, const float *  mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     S - stride step length
//     */

//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;
//     (void)H_out; // silence declared but never referenced warning. remove this line when you start working
//     (void)W_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int temp = (ceil(W_out/(1.0*TILE_WIDTH)));

//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     int h = (blockIdx.z/temp)*TILE_WIDTH+threadIdx.y;
//     int w = (blockIdx.z%temp)*TILE_WIDTH+threadIdx.x;

//     if (w<W_out) 
//     {
//         if(h<H_out)
//         {
//             if(by<M)
//             {
//                 float f = 0.0f;
//                 for (int channel=0; channel<C; channel++) 
//                 {
//                     for (int p = 0; p < K; p++) 
//                     {
//                         for (int q = 0; q < K; q++) 
//                         {
//                             f+=in_4d(bx, channel, h*S+p, w*S+q)*mask_4d(by,channel,p,q);
//                         }
//                     }
//                 }
//                 out_4d(blockIdx.x, by, h, w) = f;

//             }
//         }
//     }

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }


	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
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


//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;

//     size_t val1 = (B * M * H_out * W_out)*sizeof(float);
//     size_t val2 = (B * C * H * W)*sizeof(float);
//     size_t val3 = (M * C * K * K)*sizeof(float);

//     cudaMalloc((void **) device_output_ptr, val1);
//     cudaMalloc((void **) device_input_ptr, val2);
//     cudaMalloc((void **) device_mask_ptr, val3);

//     cudaStream_t streams[B];
//     for (int i = 0; i < B; ++i) 
//     {
//         cudaStreamCreate(&streams[i]);
//     }

//     cudaMemcpyAsync(*device_mask_ptr, host_mask, val3, cudaMemcpyHostToDevice, streams[0]);

//     dim3 dimGrid(1, M, ceil((float)W_out/TILE_WIDTH) * ceil((float)H_out/TILE_WIDTH));
//     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  
//     for(int i = 0; i < B; ++i) 
//     {
//         cudaMemcpyAsync(*device_input_ptr + i * (C*H*W), host_input + i * (C*H*W), (C*H*W) * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
//         conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(*device_output_ptr + i * (M*H_out*W_out), *device_input_ptr + i * (C*H*W), *device_mask_ptr, B, M, C, H, W, K, S);
//         cudaMemcpyAsync((float*)host_output + i * (M*H_out*W_out), (*device_output_ptr) + i * (M*H_out*W_out), (M*H_out*W_out) * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
//     }

//     for (int i = 0; i < B; ++i) 
//     {
//         cudaStreamSynchronize(streams[i]);
//         cudaStreamDestroy(streams[i]);
//     }

//     cudaFree(device_mask_ptr);
//     cudaFree(device_input_ptr);
//     cudaFree(device_output_ptr);
   
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     return;  
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {   
//     cudaFree(device_mask);
//     cudaFree(device_input);
//     cudaFree(device_output);
// }


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








