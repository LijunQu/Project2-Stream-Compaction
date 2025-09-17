/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);



    // ===== EXTRA BLOCK: DIFFERENT SCANS (prints N) =====
    {
        printf("\n");
        printf("*****************************\n");
        printf("** EXTRA SCAN TESTS **\n");
        printf("*****************************\n");

        const int potSizes[] = {
    (1 << 4),  (1 << 5),  (1 << 6),  (1 << 7),  (1 << 8),
    (1 << 9),  (1 << 10), (1 << 11), (1 << 12), (1 << 13),
    (1 << 14), (1 << 15), (1 << 16), (1 << 17), (1 << 18),
    (1 << 19), (1 << 20), (1 << 21), (1 << 22), (1 << 23),
    (1 << 24), (1 << 25), (1 << 26), (1 << 27)
        };
        const int npotSizes[] = {
    7, 13, 37, 123, 457, 1003,                    // your originals
    (1 << 4) - 1,  (1 << 5) - 3,  (1 << 6) - 5,  (1 << 7) - 11,
    (1 << 8) - 3,  (1 << 9) - 5,  (1 << 10) - 11, (1 << 11) - 3,
    (1 << 12) - 5,  (1 << 13) - 11, (1 << 14) - 3,  (1 << 15) - 5,
    (1 << 16) - 11, (1 << 17) - 3,  (1 << 18) - 5,  (1 << 19) - 11,
    (1 << 20) - 3,  (1 << 21) - 5,  (1 << 22) - 11, (1 << 23) - 3,
    (1 << 24) - 5,  (1 << 25) - 11, (1 << 26) - 3,  (1 << 27) - 5
        };


        auto runScanSuite = [&](int N) {
            int* A = new int[N], * B = new int[N], * C = new int[N];

            // Build input for scan (like your harness): random with trailing 0
            genArray(N - 1, A, 50); A[N - 1] = 0;

            // CPU reference
            zeroArray(N, B);
            { char label[128]; snprintf(label, sizeof(label), "cpu scan (N=%d)", N); printDesc(label); }
            StreamCompaction::CPU::scan(N, B, A);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

            // Naive scan
            zeroArray(N, C);
            { char label[128]; snprintf(label, sizeof(label), "naive scan (N=%d)", N); printDesc(label); }
            StreamCompaction::Naive::scan(N, C, A);
            printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
            printCmpResult(N, B, C);

            // Work-efficient scan
            zeroArray(N, C);
            { char label[128]; snprintf(label, sizeof(label), "work-efficient scan (N=%d)", N); printDesc(label); }
            StreamCompaction::Efficient::scan(N, C, A);
            printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
            printCmpResult(N, B, C);

            // Thrust scan
            zeroArray(N, C);
            { char label[128]; snprintf(label, sizeof(label), "thrust scan (N=%d)", N); printDesc(label); }
            StreamCompaction::Thrust::scan(N, C, A);
            printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
            printCmpResult(N, B, C);

            delete[] A; delete[] B; delete[] C;
            };

        for (int N : potSizes)  runScanSuite(N);
        for (int N : npotSizes) runScanSuite(N);


    }



    // ===== EXTRA BLOCK: DIFFERENT COMPACTS (prints N) =====
    {
        printf("\n");
        printf("*****************************\n");
        printf("** EXTRA COMPACTS TESTS **\n");
        printf("*****************************\n");

        const int potSizes[] = {
    (1 << 4),  (1 << 5),  (1 << 6),  (1 << 7),  (1 << 8),
    (1 << 9),  (1 << 10), (1 << 11), (1 << 12), (1 << 13),
    (1 << 14), (1 << 15), (1 << 16), (1 << 17), (1 << 18),
    (1 << 19), (1 << 20), (1 << 21), (1 << 22), (1 << 23),
    (1 << 24), (1 << 25), (1 << 26), (1 << 27)
        };
        const int npotSizes[] = {
    7, 13, 37, 123, 457, 1003,                    // your originals
    (1 << 4) - 1,  (1 << 5) - 3,  (1 << 6) - 5,  (1 << 7) - 11,
    (1 << 8) - 3,  (1 << 9) - 5,  (1 << 10) - 11, (1 << 11) - 3,
    (1 << 12) - 5,  (1 << 13) - 11, (1 << 14) - 3,  (1 << 15) - 5,
    (1 << 16) - 11, (1 << 17) - 3,  (1 << 18) - 5,  (1 << 19) - 11,
    (1 << 20) - 3,  (1 << 21) - 5,  (1 << 22) - 11, (1 << 23) - 3,
    (1 << 24) - 5,  (1 << 25) - 11, (1 << 26) - 3,  (1 << 27) - 5
        };


        auto runCompactSuite = [&](int N) {
            int* A = new int[N], * B = new int[N], * C = new int[N];

            // Build input for compaction (like your harness): values in [0,3], trailing 0
            genArray(N - 1, A, 4); A[N - 1] = 0;

            // CPU compact (baseline, ground truth)
            zeroArray(N, B);
            { char label[160]; snprintf(label, sizeof(label), "cpu compact WITHOUT scan (ref) (N=%d)", N); printDesc(label); }
            int expected = StreamCompaction::CPU::compactWithoutScan(N, B, A);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            // printArray(expected, B, true); // uncomment if you want to view contents

            // CPU compact WITH scan
            zeroArray(N, C);
            { char label[160]; snprintf(label, sizeof(label), "cpu compact WITH scan (N=%d)", N); printDesc(label); }
            int cntCpuScan = StreamCompaction::CPU::compactWithScan(N, C, A);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            printCmpLenResult(cntCpuScan, expected, B, C);

            // GPU work-efficient compact
            zeroArray(N, C);
            { char label[160]; snprintf(label, sizeof(label), "work-efficient compact (GPU) (N=%d)", N); printDesc(label); }
            int cntGpu = StreamCompaction::Efficient::compact(N, C, A);
            printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
            printCmpLenResult(cntGpu, expected, B, C);

            delete[] A; delete[] B; delete[] C;
            };

        for (int N : potSizes)  runCompactSuite(N);
        for (int N : npotSizes) runCompactSuite(N);
    }






    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
