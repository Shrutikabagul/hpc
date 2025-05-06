//cuda multiplication

#include <iostream>
#include <cuda.h>

using namespace std;

// --------------------------------------
// CUDA Kernel for Matrix Multiplication
__global__ void matrixMul(int* A, int* B, int* C, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// --------------------------------------
int main() {

    // --------------------------------------
    // Matrix Multiplication

    int N;
    cout << "\n\nEnter size N for NxN matrices (e.g. 4): ";
    cin >> N;

    int size = N * N;
    int* h_M1 = new int[size];
    int* h_M2 = new int[size];
    int* h_M3 = new int[size];

    // Initialize matrices
    for (int i = 0; i < size; i++) {
        h_M1[i] = i + 1;
        h_M2[i] = i + 1;
    }

    int *d_M1, *d_M2, *d_M3;
    cudaMalloc(&d_M1, size * sizeof(int));
    cudaMalloc(&d_M2, size * sizeof(int));
    cudaMalloc(&d_M3, size * sizeof(int));

    cudaMemcpy(d_M1, h_M1, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, h_M2, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_M1, d_M2, d_M3, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_M3, d_M3, size * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "\nMatrix Multiplication Result (first 4x4 block):\n";
    for (int i = 0; i < min(N, 4); i++) {
        for (int j = 0; j < min(N, 4); j++) {
            cout << h_M3[i * N + j] << "\t";
        }
        cout << endl;
    }

    // Free memory
    delete[] h_M1; delete[] h_M2; delete[] h_M3;
    cudaFree(d_M1); cudaFree(d_M2); cudaFree(d_M3);

    return 0;
}


/*
nvcc matrix_multiplication.cu -o matrix_multiplication
./matrix_multiplication
*/
