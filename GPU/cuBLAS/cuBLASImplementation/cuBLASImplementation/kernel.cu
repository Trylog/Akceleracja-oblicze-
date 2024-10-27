﻿#include <iostream>
#include <cublas_v2.h>
#include <curand.h>
#include <random>

using namespace std;

void loadDimensions(int& m, int& n, int& k) {
	cout << "Give 1. matrix's row size" << endl;
	cout << "m: ";
	cin >> m;
	cout << endl << "Give 1. matrix's column size and 2. matrix's row size" << endl;
	cout << "n: ";
	cin >> n;
	cout << endl << "Matrix A dimensions: " << m << " X " << n << endl;

	cout << endl << "Give 2. matrix's column size" << endl;
	cout << "k: ";
	cin >> k;
	cout << endl << "Matrix B dimensions: " << n << " X " << k << endl;
}

template <typename T>
void printMatrixColumnMajorOrder(T* M, int a, int b, string matrixName) {
	//print read matrix
	cout << endl << "Matrix " + matrixName + ": " << endl;

	for (int i = 0; i < a; i++) {
		for (int j = 0; j < b; j++) {
			cout << M[i + j * a] << ", ";
		}
		cout << endl;
	}
}

template <typename T>
void randomizeMatrices(int m, int n, int k, T* A, T* B, T* C) {
	//Generating matrices

	//1. Init generator
	//arguments: (pointer to generator, type of generator to create)
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

	//2. Set generator options (seed, offset, order)
	//arguments: (generator, seed)
	curandSetPseudoRandomGeneratorSeed(generator, clock());

	//3. Generate random numbers

	if constexpr (std::is_same<T, float>::value) {
		curandGenerateUniform(generator, A, m * n);
		curandGenerateUniform(generator, B, n * k);
	}
	else if constexpr (std::is_same<T, double>::value) {
		curandGenerateUniformDouble(generator, A, m * n);
		curandGenerateUniformDouble(generator, B, n * k);
	}
	else {
		// If T is something else
		std::cout << "Integer" << std::endl;
	}

	// Scaling range to [-10, 10]
	T min = -10;
	T max = 10;
	T scale = max - min;

	// Scale each element in d_A and d_B
	cublasHandle_t handle;
	cublasCreate(&handle);

	if constexpr (std::is_same<T, float>::value) {
		// First, multiply each element by `scale`
		cublasSscal(handle, m * n, &scale, A, 1);
		cublasSscal(handle, n * k, &scale, B, 1);

		// Then, add `min` to shift the values
		cublasSaxpy(handle, m * n, &min, A, 1, A, 1);
		cublasSaxpy(handle, n * k, &min, B, 1, B, 1);
	}
	else if constexpr (std::is_same<T, double>::value) {
		// First, multiply each element by `scale`
		cublasDscal(handle, m * n, &scale, A, 1);
		cublasDscal(handle, n * k, &scale, B, 1);

		// Then, add `min` to shift the values
		cublasDaxpy(handle, m * n, &min, A, 1, A, 1);
		cublasDaxpy(handle, n * k, &min, B, 1, B, 1);
	}
	else {
		//For integers, there's no method to generate numbers in cublas library, so classicaly:
		for (int i = 0; i < m * n; i++) {
			A[i] = (rand() % (max - min + 1)) + min;
		}

		for (int i = 0; i < n * k; i++) {
			B[i] = (rand() % (max - min + 1)) + min;
		}
	}

	//4. Cleanup
	curandDestroyGenerator(generator);

	//input matrices are stored in column-major order
	printMatrixColumnMajorOrder(A, m, n, "A");
	printMatrixColumnMajorOrder(B, n, k, "B");
}

template <typename T>
void fixedMatrices(int m, int n, int k, T* A, T* B) {
	//column-major order
	int number = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			A[i + j * m] = number++;
		}
	}

	number = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			B[i + j * n] = number++;
		}
	}

	//input matrices are stored in column-major order
	printMatrixColumnMajorOrder(A, m, n, "A");
	printMatrixColumnMajorOrder(B, n, k, "B");
}

//generic
template <typename T>
void program() {
	//matrix dimensions
	int m, n;
	int k;

	loadDimensions(m, n, k);

	//Allocate memory
	//two matrices
	T* A, * B;

	//score matrix in GPU memory
	T* C;

	//score matrix in native memory
	T* D;

	cudaMallocHost(&A, m * n * sizeof(T));
	cudaMallocHost(&B, n * k * sizeof(T));
	cudaMallocHost(&C, m * k * sizeof(T));
	cudaMallocHost(&D, m * k * sizeof(T));

	//random matrices or fixed matrices defined by user in code
	int option;
	cout << endl << "1. Random matrices" << endl;
	cout << "2. Own matrices defined in code" << endl;
	cin >> option;

	if (option == 1) {
		randomizeMatrices<T>(m, n, k, A, B, C);
	}
	else {
		fixedMatrices<T>(m, n, k, A, B);
	}

	//Multiplication operation
	cublasHandle_t handle;
	cublasCreate(&handle);

	T alpha = 1;
	T beta = 0;

	//find leading dimensions of matrices
	int lda = m;
	int ldb = n;
	int ldc = m;

	//measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//start event
	cudaEventRecord(start, 0);

	//CUBLAS_OP_N - non-transpose operation
	//cublasSgemm(h,transpA,transpB,m,k,n,&alpha,&A,lda,&B,ldb,&beta,&C,ldc)

	if constexpr (std::is_same<T, float>::value) {
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
	}
	else if constexpr (std::is_same<T, double>::value) {
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
	}
	else {
		cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I, ldb, &beta, C, CUDA_R_8I, ldc, CUDA_R_8I, CUBLAS_GEMM_DEFAULT);
	}

	//retrieve matrix from gpu memory
	//cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
	cublasGetMatrix(m, k, sizeof(T), C, ldc, D, ldc);

	//stop event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << endl << "It took: " << elapsedTime << " seconds" << endl;

	//output matrix is stored in column-major order
	printMatrixColumnMajorOrder<T>(C, m, k, "score");

	//free up memory
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(D);
	cublasDestroy(handle);
}

int main() {
	int dataType;
	cout << "Choose data type: " << endl;
	cout << "1. Float" << endl;
	cout << "2. Integer" << endl;
	cout << "3. Double" << endl;
	cin >> dataType;

	switch (dataType) {
	case 1: program<float>(); break;
	case 2: program<int>(); break;
	case 3: program<double>(); break;
	}
}