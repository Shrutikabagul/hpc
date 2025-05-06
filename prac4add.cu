// cuda addition

#include <iostream>
#include <cuda.h>

using namespace std;

// --------------------------------------
// CUDA Kernel for Vector Addition
__global__ void vectorAdd(int* A, int* B, int* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

// --------------------------------------
int main() {
    int n;
    cout << "Enter size of vectors (e.g. 1024): ";
    cin >> n;

    // Host memory allocation for vectors
    int* h_A = new int[n];
    int* h_B = new int[n];
    int* h_C = new int[n];

    // Initialize vectors
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // Device memory allocation
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, n * sizeof(int));

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch vector addition kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "\nVector Addition Result (first 10 values): ";
    for (int i = 0; i < min(10, n); i++) {
        cout << h_C[i] << " ";
    }
    cout << endl;

    // Free memory
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
