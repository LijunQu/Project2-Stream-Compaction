#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            if (n <= 0 || !odata || !idata) {
                timer().endCpuTimer();
                return;
            }

            int sum = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = sum;
                sum += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            if (n <= 0 || !odata || !idata) {
                timer().endCpuTimer();
                return -1;
            }

            int cnt = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[cnt] = idata[i];
                    ++cnt;
                }
            }
            timer().endCpuTimer();
            return cnt;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */

        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            if (n <= 0 || !odata || !idata) {
                timer().endCpuTimer();
                return -1;
            }

            int cnt = 0;

            // stream compaction using the scan function. 
            // Map the input array to an array of 0s and 1s, 
            // scan it, and use scatter to produce the output. 
            // You will need a CPU scatter implementation for this 
            // (see slides or GPU Gems chapter for an explanation).


            int* flag = new int[n];
            int* scan = new int[n];

            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) ++cnt;
                flag[i] = (idata[i] != 0) ? 1 : 0;
            }

            scan[0] = 0;
            for (int i = 1; i < n; ++i) {
                scan[i] = scan[i - 1] + flag[i - 1];
            }

            for (int i = 0; i < n; ++i) {
                odata[scan[i]] = idata[i];
            }


            delete[] flag;
            delete[] scan;

            timer().endCpuTimer();
            return cnt;
        }
    }
}
