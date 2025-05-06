#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA kernel to add two vectors
__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main() {
    int N;
    cout << "Enter the size of the vectors: ";
    cin >> N;

    // Allocate host memory
    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    // Input vector A
    cout << "Enter elements of vector A:\n";
    for (int i = 0; i < N; i++) cin >> h_A[i];

    // Input vector B
    cout << "Enter elements of vector B:\n";
    for (int i = 0; i < N; i++) cin >> h_B[i];

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with enough blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Output result
    cout << "Result of vector addition:\n";
    for (int i = 0; i < N; i++) cout << h_C[i] << " ";
    cout << endl;

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
/*1. Save as vector_add.cu
nvcc vector_add.cu -o vector_add
./vector_add
Enter the size of the vectors: 4
Enter elements of vector A:
1 2 3 4
Enter elements of vector B:
5 6 7 8
Result of vector addition:
6 8 10 12

*/
