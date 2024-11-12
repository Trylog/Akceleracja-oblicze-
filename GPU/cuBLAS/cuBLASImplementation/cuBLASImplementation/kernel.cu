#include <iostream>
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
			if (std::is_same<T, int8_t>::value) {
				cout << static_cast<int32_t>(M[i + j * a]);
			}
			else {
				cout << M[i + j * a];
			}
			if (j < b - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}
}

template <typename T>
void randomizeMatrices(int m, int n, int k, T* A, T* B) {
	
	//Generating matrices
	//float, double
	if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
		//1. Init generator
		//arguments: (pointer to generator, type of generator to create)
		curandGenerator_t generator;
		curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

		//2. Set generator options (seed, offset, order)
		//arguments: (generator, seed)
		curandSetPseudoRandomGeneratorSeed(generator, clock());

		//3. Generate random numbers
		//float
		if constexpr (std::is_same<T, float>::value) {
			curandGenerateUniform(generator, A, m * n);
			curandGenerateUniform(generator, B, n * k);
		}
		//double
		else {
			curandGenerateUniformDouble(generator, A, m * n);
			curandGenerateUniformDouble(generator, B, n * k);
		}

		// Scaling range to [-10, 10]
		T min = -128;
		T max = 127;
		T scale = max - min;

		// Scale each element in d_A and d_B
		cublasHandle_t handle;
		cublasCreate(&handle);

		//float
		if constexpr (std::is_same<T, float>::value) {
			// First, multiply each element by `scale`
			cublasSscal(handle, m * n, &scale, A, 1);
			cublasSscal(handle, n * k, &scale, B, 1);

			// Then, add `min` to shift the values
			cublasSaxpy(handle, m * n, &min, A, 1, A, 1);
			cublasSaxpy(handle, n * k, &min, B, 1, B, 1);
		}
		//double
		else {
			// First, multiply each element by `scale`
			cublasDscal(handle, m * n, &scale, A, 1);
			cublasDscal(handle, n * k, &scale, B, 1);

			// Then, add `min` to shift the values
			cublasDaxpy(handle, m * n, &min, A, 1, A, 1);
			cublasDaxpy(handle, n * k, &min, B, 1, B, 1);
		}

		//4. Cleanup
		curandDestroyGenerator(generator);

		//input matrices are stored in column-major order
		printMatrixColumnMajorOrder(A, m, n, "A");
		printMatrixColumnMajorOrder(B, n, k, "B");
	}

	//For integers, there's no method to generate numbers in curand library, so classicaly:
	else {
		srand(static_cast<unsigned int>(time(NULL)));
		//matrix dimensions have to be multiplication of 4
		int closestM = ((m + 3) / 4) * 4;
		int closestN = ((n + 3) / 4) * 4;

		for (int i = 0; i < closestM; i++) {
			for (int j = 0; j < closestN; j++) {
				if (i >= m || j >= n) {
					A[i * closestM + j] = 0;
				}
				else {
					A[i * closestM + j] = static_cast<T>(rand() % (256 - 128));
				}
			}
		}

		int closestK = ((k + 3) / 4) * 4;
		for (int i = 0; i < closestN; i++) {
			for (int j = 0; j < closestK; j++) {
				if (i >= n || j >= k) {
					B[i * closestN + j] = 0;
				}
				else {
					B[i * closestN + j] = static_cast<T>(rand() % (256 - 128));
				}
			}
		}
		//input matrices are stored in column-major order
		printMatrixColumnMajorOrder(A, closestM, closestN, "A");
		printMatrixColumnMajorOrder(B, closestN, closestK, "B");
	}
}

int* removePadding(int* M, int closestM, int closestK, int m, int k) {
	int* score = new int[m * k];

	for (int i = 0; i < closestM; i++) {
		for (int j = 0; j < closestK; j++) {
			if (i < m || j < k) {
				score[i * closestM + j] = M[i * closestM + j];
			}
		}
	}

	return score;
}

template <typename T, typename U>
void distinguish(int m, int n, int k, T* A, T* B) {
	U* C;
	//score matrix in native memory
	U* D;

	if (std::is_same<T, double>::value || std::is_same<T, float>::value) {
		cudaMallocHost(&C, m * k * sizeof(U));
		cudaMallocHost(&D, m * k * sizeof(U));
	}
	else {
		int closestM = ((m + 3) / 4) * 4;
		int closestK = ((k + 3) / 4) * 4;

		cudaMallocHost(&C, closestM * closestK * sizeof(U));
		cudaMallocHost(&D, closestM * closestK * sizeof(U));
	}
	

	randomizeMatrices<T>(m, n, k, A, B);

	//Multiplication operation
	cublasHandle_t handle;
	cublasCreate(&handle);

	U alpha = 1;
	U beta = 0;

	//CUBLAS_OP_N - non-transpose operation
	//cublasSgemm(h,transpA,transpB,m,k,n,&alpha,&A,lda,&B,ldb,&beta,&C,ldc)

	if constexpr (std::is_same<U, float>::value || std::is_same<U, double>::value) {
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

		//float
		if constexpr (std::is_same<U, float>::value) {
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
		}

		//double
		else {
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
		}

		//retrieve matrix from gpu memory
		//cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
		cublasGetMatrix(m, k, sizeof(U), C, ldc, D, ldc);

		//stop event
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cout << endl << "It took: " << elapsedTime << " milliseconds" << endl;

		//output matrix is stored in column-major order
		printMatrixColumnMajorOrder<U>(C, m, k, "score");
	}

	else {
		int closestM = ((m + 3) / 4) * 4;
		int closestN = ((n + 3) / 4) * 4;

		int closestK = ((k + 3) / 4) * 4;

		//find leading dimensions of matrices
		int lda = closestM;
		int ldb = closestN;
		int ldc = closestM;

		//measure time
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//start event
		cudaEventRecord(start, 0);
		cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, closestM, closestK, closestN, &alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I, ldb, &beta, C, CUDA_R_32I, ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

		//retrieve matrix from gpu memory
		//cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
		cublasGetMatrix(closestM, closestK, sizeof(U), C, ldc, D, ldc);

		//stop event
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cout << endl << "It took: " << elapsedTime << " milliseconds" << endl;

		//output matrix is stored in column-major order
		int * score = removePadding(C, closestM, closestK, m, k);
		printMatrixColumnMajorOrder<U>(score, m, k, "score");
	}

	//free up memory
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(D);
	cublasDestroy(handle);
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

	if (std::is_same<T, double>::value || std::is_same<T, float>::value) {
		cudaMallocHost(&A, m * n * sizeof(T));
		cudaMallocHost(&B, n * k * sizeof(T));
	}
	//int8_t, dimensions have to be multiplication of 4
	else {
		int closestM = ((m + 3) / 4) * 4;
		int closestN = ((n + 3) / 4) * 4;

		int closestK = ((k + 3) / 4) * 4;

		cudaMallocHost(&A, closestM * closestN * sizeof(T));
		cudaMallocHost(&B, closestN * closestK * sizeof(T));
	}
	

	//score matrix in GPU memory
	//for 8bit integers - score is 32 bits
	if constexpr (std::is_same<T, int8_t>::value) {
		distinguish<T, int32_t>(m, n, k, A, B);
	}
	else {
		distinguish<T, T>(m, n, k, A, B);
	}
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
	case 2: program<int8_t>(); break;
	case 3: program<double>(); break;
	}
}