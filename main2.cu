//Michał Bernacki-Janson
//Matrix Multiplication on CUDA, simplest version

#include <iostream>
#include <cuda.h>

using namespace std;

#define BLOCK_SIZE 16

template <typename T>
__global__ void cuda_hello(T* tab1, T* tab2, T* tab3, int x_size, int y_size) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < x_size && y < y_size) {
        T sum = 0;
        for (int i = 0; i < x_size; ++i) {
            sum += tab1[y * x_size + i] * tab2[i * y_size + x];
        }
        tab3[y * x_size + x] = sum;
    }
}

template <typename T>
void matrixMultiply(int N, int M) {
    int size = N * M * sizeof(T);
    auto h_tab1 = (T*) malloc(size);
    auto h_tab2 = (T*) malloc(size);
    auto h_tab3 = (T*) malloc(size);

    for (int i = 0; i < N * M; i++) {
        h_tab1[i] = 1;
        h_tab2[i] = 1;
        h_tab3[i] = 0;
    }

    T* d_tab1;
    T* d_tab2;
    T* d_tab3;

    cudaMalloc(&d_tab1, size);
    cudaMalloc(&d_tab2, size);
    cudaMalloc(&d_tab3, size);

    cudaMemcpy(d_tab1, h_tab1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tab2, h_tab2, size, cudaMemcpyHostToDevice);

    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    cuda_hello<T><<<dimGrid, dimBlock>>>(d_tab1, d_tab2, d_tab3, N, M);

    cudaDeviceSynchronize();

    cudaMemcpy(h_tab3, d_tab3, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << h_tab3[i * N + j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_tab1);
    cudaFree(d_tab2);
    cudaFree(d_tab3);

    free(h_tab1);
    free(h_tab2);
    free(h_tab3);
}

int main() {
    const int N = 10;
    const int M = 10;

    cout << "Matrix multiplication with floats:" << endl;
    matrixMultiply<float>(N, M);

    cout << "Matrix multiplication with doubles:" << endl;
    matrixMultiply<double>(N, M);

    return 0;
}
