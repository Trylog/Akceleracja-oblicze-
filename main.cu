//Michał Bernacki-Janson
//Matrix Multiplication on CUDA using shared memory

#include <iostream>
#include <cuda.h>

using namespace std;

#define BLOCK_SIZE 16

template <typename T>
struct Matrix {
    T* elements;
    int width;
    int height;
    int stride;
};

template <typename T>
__device__ T getElement(const Matrix<T> A, int row, int col) {
    return A.elements[row * A.stride + col];
}

template <typename T>
__device__ void setElement(Matrix<T> A, int row, int col, T value) {
    A.elements[row * A.stride + col] = value;
}

template <typename T>
__device__ Matrix<T> getSubMatrix(Matrix<T> A, int row, int col) {
    Matrix<T> Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

template <typename T>
__global__ void matMulKernel(const Matrix<T> A, const Matrix<T> B, Matrix<T> C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix<T> Csub = getSubMatrix(C, blockRow, blockCol);
    T Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    bool retFlag = false;

    int globalRow = blockRow * BLOCK_SIZE + row;
    int globalCol = blockCol * BLOCK_SIZE + col;
    if (globalRow >= C.height || globalCol >= C.width) { // ZMIANA
        retFlag = true;
    }

    for (int m = 0; m < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        Matrix<T> Asub = getSubMatrix(A, blockRow, m);
        Matrix<T> Bsub = getSubMatrix(B, m, blockCol);

        __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = (row < Asub.height && col < Asub.width) ? getElement(Asub, row, col) : 0;
        Bs[row][col] = (row < Bsub.height && col < Bsub.width) ? getElement(Bsub, row, col) : 0;

        __syncthreads();

        if (!retFlag) {
            for (int e = 0; e < BLOCK_SIZE; ++e){
                Cvalue += As[row][e] * Bs[e][col];
            }
        }

        __syncthreads();
    }

    if (retFlag) return;

    if (row < Csub.height && col < Csub.width) {
        setElement(Csub, row, col, Cvalue);
    }
}

template <typename T>
void matMul(const Matrix<T> A, const Matrix<T> B, Matrix<T> C) {
    Matrix<T> d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(T);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix<T> d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(T);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    Matrix<T> d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(T);
    cudaMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + BLOCK_SIZE - 1) / BLOCK_SIZE, (A.height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

int main() {
    const int N = 33;
    const int M = 33;

    Matrix<float> A, B, C;
    A.width = B.width = C.width = N;
    A.height = B.height = C.height = M;
    A.stride = B.stride = C.stride = C.width;

    size_t size = N * M * sizeof(float);
    cudaMallocHost(&A.elements, size);
    cudaMallocHost(&B.elements, size);
    cudaMallocHost(&C.elements, size);

    for (int i = 0; i < N * M; i++) {
        A.elements[i] = 1.0f;
        B.elements[i] = 1.0f;
        C.elements[i] = 0.0f;
    }

    matMul(A, B, C);

    cout << "Result Matrix C (float):" << endl;
    for (int i = 0; i < C.height; i++) {
        for (int j = 0; j < C.width; j++) {
            cout << C.elements[i * C.stride + j] << " ";
        }
        cout << endl;
    }

    cudaFreeHost(A.elements);
    cudaFreeHost(B.elements);
    cudaFreeHost(C.elements);

    Matrix<double> A_double, B_double, C_double;
    A_double.width = B_double.width = C_double.width = N;
    A_double.height = B_double.height = C_double.height = M;
    A_double.stride = B_double.stride = C_double.stride = C_double.width;

    size_t size_double = N * M * sizeof(double);
    cudaMallocHost(&A_double.elements, size_double);
    cudaMallocHost(&B_double.elements, size_double);
    cudaMallocHost(&C_double.elements, size_double);

    for (int i = 0; i < N * M; i++) {
        A_double.elements[i] = 1.0;
        B_double.elements[i] = 1.0;
        C_double.elements[i] = 0.0;
    }

    matMul(A_double, B_double, C_double);

    cout << "Result Matrix C (double):" << endl;
    for (int i = 0; i < C_double.height; i++) {
        for (int j = 0; j < C_double.width; j++) {
            cout << C_double.elements[i * C_double.stride + j] << " ";
        }
        cout << endl;
    }

    cudaFreeHost(A_double.elements);
    cudaFreeHost(B_double.elements);
    cudaFreeHost(C_double.elements);

    return 0;
}
