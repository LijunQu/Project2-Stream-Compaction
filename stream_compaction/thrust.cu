#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());


            thrust::host_vector<int> host_idata(idata, idata + n);

            thrust::device_vector<int> dev_idata(host_idata);
            thrust::device_vector<int> dev_odata(n);
            cudaDeviceSynchronize();



            thrust::device_vector<int> d_in(host_idata); 
            thrust::device_vector<int> d_out(n);
            cudaDeviceSynchronize();

            timer().startGpuTimer();
            thrust::exclusive_scan(d_in.begin(), d_in.end(), d_out.begin());
            timer().endGpuTimer();

            thrust::copy(d_out.begin(), d_out.end(), odata);

            cudaDeviceSynchronize();



            
        }
    }
}
