#include <iostream>
#include <cublas_v2.h>
#include <curand.h>

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

void randomizeMatrices(int m, int n, int k, float* A, float* B, float* C) {
	//Generating matrices

	//1. Init generator
	//arguments: (pointer to generator, type of generator to create)
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

	//2. Set generator options (seed, offset, order)
	//arguments: (generator, seed)
	curandSetPseudoRandomGeneratorSeed(generator, 1234UL);

	//3. Generate random numbers
	curandGenerateUniform(generator, A, m * n);
	curandGenerateUniform(generator, B, n * k);

	// Scaling range to [-10, 10]
	float min = -10.0f;
	float max = 10.0f;
	float scale = max - min;

	// Scale each element in d_A and d_B
	cublasHandle_t handle;
	cublasCreate(&handle);

	// First, multiply each element by `scale`
	cublasSscal(handle, m * n, &scale, A, 1);
	cublasSscal(handle, n * k, &scale, B, 1);

	// Then, add `min` to shift the values
	cublasSaxpy(handle, m * n, &min, A, 1, A, 1);
	cublasSaxpy(handle, n * k, &min, B, 1, B, 1);

	//4. Cleanup
	curandDestroyGenerator(generator);


	//Check matrices which were generated
	float* generatedMatrixA = (float*)malloc(m * n * sizeof(float));
	float* generatedMatrixB = (float*)malloc(n * k * sizeof(float));

	cudaMemcpy(generatedMatrixA, A, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(generatedMatrixB, B, n * k * sizeof(float), cudaMemcpyDeviceToHost);

	//print read matrix
	cout << endl << "Matrix A:" << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			cout << A[i * n + j] << ", ";
		}
		cout << endl;
	}

	//print read matrix
	cout << endl << "Matrix B:" << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			cout << B[i * k + j] << ", ";
		}
		cout << endl;
	}
}

void fixedMatrices(int m, int n, int k, float* A, float* B) {
	//row-major order
	for (int i = 0; i < m * n; i++) {
		A[i] = i;
	}

	for (int i = 0; i < n * k; i++) {
		B[i] = i;
	}

	//print read matrix
	cout << endl << "Matrix A:" << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			cout << A[i * n + j] << ", ";
		}
		cout << endl;
	}

	//print read matrix
	cout << endl << "Matrix B:" << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			cout << B[i * k + j] << ", ";
		}
		cout << endl;
	}
}

int main() {

	//matrix dimensions
	int m, n;
	int k;

	loadDimensions(m, n, k);

	//Allocate memory
	//two matrices
	float* A, * B;

	//score matrix in GPU memory
	float* C;

	//score matrix in native memory
	float* D;

	cudaMallocHost(&A, m * n * sizeof(float));
	cudaMallocHost(&B, n * k * sizeof(float));
	cudaMallocHost(&C, m * k * sizeof(float));
	cudaMallocHost(&D, m * k * sizeof(float));

	//random matrices or fixed matrices defined by user in code
	int option;
	cout << endl << "1. Random matrices" << endl;
	cout << "2. Own matrices defined in code" << endl;
	cin >> option;

	if (option == 1) {
		randomizeMatrices(m, n, k, A, B, C);
	}
	else {
		fixedMatrices(m, n, k, A, B);
	}

	//Multiplication operation
	cublasHandle_t handle;
	cublasCreate(&handle);

	float alpha = 1.0f;
	float beta = 0.0f;

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
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, A, lda, B, ldb, &beta, C, ldc);

	//retrieve matrix from gpu memory
	//cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
	cublasGetMatrix(m, k, sizeof(float), C, ldc, D, ldc);

	//stop event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << endl << "It took: " << elapsedTime << " seconds" << endl;

	//print score matrix
	//row-major order
	cout << "Score matrix:" << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k; j++) {
			cout << D[i * k + j] << ", ";
		}
		cout << endl;
	}
}