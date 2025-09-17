#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
using namespace StreamCompaction::Common;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __host__ __device__ inline int nextPow2(int x) {
            int p = 1;
            while (p < x) p <<= 1;
            return p;
        }

        __host__ __device__ inline int ilog2(int x) {
            int r = 0;
            while ((1 << (r + 1)) <= x) ++r;
            return r;
        }
        __global__ void kernUpSweepStrided(int n2, int d, int* data, int work) {
            int offset = 1 << d;
            int stride = offset << 1;
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int step = blockDim.x * gridDim.x;
            for (int t = tid; t < work; t += step) {
                int i = (t + 1) * stride - 1;
                data[i] += data[i - offset];
            }
        }

        __global__ void kernDownSweepStrided(int n2, int d, int* data, int work) {
            int offset = 1 << d;
            int stride = offset << 1;
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int step = blockDim.x * gridDim.x;
            for (int t = tid; t < work; t += step) {
                int i = (t + 1) * stride - 1;
                int left = i - offset;
                int tmp = data[left];
                data[left] = data[i];
                data[i] += tmp;
            }
        }

        __global__ void kernSetLastZero(int* data, int lastIdx) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                data[lastIdx] = 0;
            }
        }



        __global__ void kernFlag(int n, const int* data, int* flag) 
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) flag[i] = (data[i] != 0);
        }


        __global__ void kernCompact(int n, int* idata, int* odata, int* flag) 
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= n) return;
            if (idata[i] != 0) {
                int dst = flag[i];
                odata[dst] = idata[i];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {

            
            if (n <= 0 || !odata || !idata) { timer().endGpuTimer(); return; }

            const int BLOCK_SIZE = 128;
            int n2 = nextPow2(n);

            int* d = nullptr;
            cudaMalloc(&d, n2 * sizeof(int));
            cudaMemcpy(d, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (n2 > n) cudaMemset(d + n, 0, (n2 - n) * sizeof(int));

            // upsweep
            timer().startGpuTimer();

            int levels = ilog2(n2);
            for (int dlev = 0; dlev < levels; ++dlev) {
                int work = n2 >> (dlev + 1); 
                int blocks = min(65535, (work + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweepStrided << <blocks, BLOCK_SIZE >> > (n2, dlev, d, work);
            }


            kernSetLastZero << <1, 1 >> > (d, n2 - 1);

            // downsweep
            for (int dlev = levels - 1; dlev >= 0; --dlev) {
                int work = n2 >> (dlev + 1);
                int blocks = min(65535, (work + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownSweepStrided << <blocks, BLOCK_SIZE >> > (n2, dlev, d, work);
            }
            timer().endGpuTimer();


            cudaMemcpy(odata, d, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d);

            cudaDeviceSynchronize();
            
        }


        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            using StreamCompaction::Common::PerformanceTimer;

            if (n <= 0 || !odata || !idata) return -1;

            const int BLOCK_SIZE = 128; 
            int n2 = nextPow2(n);
            int fullBlocksPerGrid = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // --- device buffers ---
            int* din = nullptr;
            int* dout = nullptr;
            int* flag = nullptr;

            cudaMalloc(&din, n2 * sizeof(int));
            cudaMalloc(&dout, n2 * sizeof(int));
            cudaMalloc(&flag, n2 * sizeof(int));

            cudaMemcpy(din, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (n2 > n) cudaMemset(din + n, 0, (n2 - n) * sizeof(int));
            cudaMemset(dout, 0, n2 * sizeof(int));
            cudaMemset(flag, 0, n2 * sizeof(int));


            timer().startGpuTimer();

            kernFlag << <fullBlocksPerGrid, BLOCK_SIZE >> > (n2, din, flag);

            int lastMask = 0;
            cudaMemcpy(&lastMask, flag + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            int levels = ilog2(n2);


            for (int dlev = 0; dlev < levels; ++dlev) {
                int work = n2 >> (dlev + 1);
                int blocks = min(65535, (work + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweepStrided << <blocks, BLOCK_SIZE >> > (n2, dlev, flag, work);
            }

            kernSetLastZero << <1, 1 >> > (flag, n2 - 1);

            for (int dlev = levels - 1; dlev >= 0; --dlev) {
                int work = n2 >> (dlev + 1);
                int blocks = min(65535, (work + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownSweepStrided << <blocks, BLOCK_SIZE >> > (n2, dlev, flag, work);
            }

            int lastIdx = 0;
            cudaMemcpy(&lastIdx, flag + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIdx + lastMask;

            kernCompact << <fullBlocksPerGrid, BLOCK_SIZE >> > (n2, din, dout, flag);

            timer().endGpuTimer();

            if (count > 0) {
                cudaMemcpy(odata, dout, count * sizeof(int), cudaMemcpyDeviceToHost);
            }

            cudaFree(din);
            cudaFree(dout);
            cudaFree(flag);

            return count;
        }


    }
}
