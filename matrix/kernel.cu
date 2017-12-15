
/*
	CUDA libraries
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "matrix.h" // the class representing matrix along with the multipications
#define CUDA_BLOCK 16 // cuda GPU block size, for CUDA multithreadding

#define RATIO_SECOND_TO_MS 0.001 // to convert ms (from cuda events) to seconds (human vision)



int main()
{
	int mat_size = MAX_MAT_SIZE;

	/*
		matrixes to multiply
		their size is N*N which will be MAX_MAT_SIZE from matrix.h
	*/
	matrix m1(mat_size);
	matrix m2(mat_size);

	cudaEvent_t device_begin, device_end; // calculate function processing time

	/*
		create event recorders -- should call Destroyer...
	*/
	cudaEventCreate(&device_begin); 
	cudaEventCreate(&device_end);
	clock_t start;
	clock_t end;
	/*
		Satistics variables
	*/
	float GPU_elapsed = 0, CPU_multithreadded_elapsed = 0;// CPU_single_elapsed = 0;

	int i, j; // iterators, used for random variables looping through both matrixes
	/*
		init mat1 and mat2 stage
	*/
	for (i = 0; i < mat_size; i++) // init matrix
	{
		/*
			random number is between 0 and RAND_MAX at matrix.h
		*/
		for (j = 0; j < mat_size; j++)
		{
			m1[i][j] = random_number(); // random number for matrix m1 at index i,j; 
			m2[i][j] = random_number(); // random number for matrix m2 at index i,j;
		}
	}


	/*
		CPU single core mult
	*/
	/*
	start = clock(); // parallel CPU clock counting, just in case // switched to main
	cudaEventRecord(device_begin, 0); // cpu single start recording
	matrix m3_cpu_single = m1 * m2; // cpu single core mult
	cudaEventRecord(device_end, 0); // cpu single stop recording
	cudaEventSynchronize(device_end); // sync
	end = clock(); // parallel CPU clock counting, just in case // switched to main
	CPU_single_elapsed = (end - start) / CLOCKS_PER_SEC;
	if (CPU_single_elapsed == 0)
	{
		cudaEventElapsedTime(&CPU_single_elapsed, device_begin, device_end); // calculate
		CPU_single_elapsed *= RATIO_SECOND_TO_MS; // convert to second from ms
	}
	//satistics
	std::cout << " Host (CPU single threadded) calculated matrix multipication at size of " << mat_size << "X" << mat_size << " at " << CPU_single_elapsed << " Seconds" << std::endl;
	*/

	/*
		CPU multicore mult
	*/
	start = clock(); // parallel CPU clock counting, just in case // switched to main
	cudaEventRecord(device_begin, 0); // cpu multi start recording
	matrix m3_cpu_multi = m1 * m2; // cpu multicore mult
	cudaEventRecord(device_end, 0); // cpu multi stop recording
	cudaEventSynchronize(device_end); // sync
	end = clock(); // parallel CPU clock counting, just in case // switched to main
	CPU_multithreadded_elapsed =(float) ( (end - start) / CLOCKS_PER_SEC);
	if (CPU_multithreadded_elapsed == 0)
	{
		cudaEventElapsedTime(&CPU_multithreadded_elapsed, device_begin, device_end); // calculate
		CPU_multithreadded_elapsed *= (float) RATIO_SECOND_TO_MS; // convert to second from ms
	}
	//sattistics
	std::cout << " Host (CPU Multithreadded!) calculated matrix multipication at size of " << mat_size << "X" << mat_size << " at " << CPU_multithreadded_elapsed << " Seconds" << std::endl;

	
	
	/*
				CUDA mult
	*/

	cudaEventRecord(device_begin, 0); // start recording
	matrix m3_gpu = m1.cuda_mult(m2); // cuda mult
	cudaEventRecord(device_end, 0); // stop recording
	cudaEventSynchronize(device_end); // sync
	cudaEventElapsedTime(&GPU_elapsed, device_begin, device_end); // calculate
	GPU_elapsed *= (float) RATIO_SECOND_TO_MS; // convert to second from ms
	//satisitcs
	std::cout << " Device (GPU) calculated matrix multipication at size of " << mat_size << "X" << mat_size << " at " << GPU_elapsed << " Seconds" << std::endl;


	cudaEventDestroy(device_begin); // destroy recorders
	cudaEventDestroy(device_end);
	
	/*
	    matrix printers, for testing
	*/

	/*
	std::cout << "CPU Single: " << m3_cpu_single << std::endl << std::endl;
	std::cout << "CPU Multi: " << m3_cpu_multi << std::endl << std::endl;
	std::cout << "GPU: "       << m3_gpu << std::endl << std::endl;
	*/


	/* 
			satistics printing!
	*/
	std::cout << std::endl;
	std::cout << std::endl;

	/*
			Correctness
	*/
	std::cout << "And both matrixes are ";
	if (m3_cpu_multi == m3_gpu)
		std::cout << " Equal, therefore correct" << std::endl;
	else
		std::cout << " Unequal, therefore (most likely GPU) is incorrect " << std::endl;

	std::cout << std::endl;
	std::cout << std::endl;

	/*
			RATIO - GPU and CPU multi-thread
	*/

	std::cout << " The ratio between GPU and CPU (multi_threadded) (GPU/CPU Multi-threadded) is  ";
	if (CPU_multithreadded_elapsed == 0)
		std::cout << " Uncountable";
	else
		std::cout << (GPU_elapsed) / (CPU_multithreadded_elapsed) << std::endl;


	/*

	/*
				ratio - CPU single vs CPU multi 
	/*

	std::cout << std::endl;
	std::cout << std::endl;

	std::cout << " Also, ratio between CPU multicore and CPU single core (CPU_MULTI/CPU_SINGLE) ";

	if (CPU_single_elapsed == 0)
		std::cout << " Uncountable";
	else
		std::cout << (CPU_multithreadded_elapsed) / (CPU_single_elapsed) << std::endl;

	std::cout << std::endl;
	std::cout << std::endl;

	
			ratio - GPU vs CPU single core
	

	std::cout << " Also, ratio between GPU and CPU single core (GPU/CPU_SINGLE) ";

	if (CPU_single_elapsed == 0)
		std::cout << " Uncountable";
	else
		std::cout << (GPU_elapsed) / (CPU_single_elapsed) << std::endl;
	*/
	return 0;
}



__global__ void cuda_matrix_mult(int* m1, int* m2, int* m3, int size)
{
	int row = threadIdx.x;
	int col = blockIdx.x;
	if (row >= size || col >= size)
		return;
	for (int i = 0; i < size; i++)
	{
		m3[row*size + col] += m1[row*size + i] * m2[i*size + col];
	}
}


matrix matrix::cuda_mult(const matrix &other)
{
	matrix matrix3(this->N); // store results here
	if (this->N != other.N)
		return matrix3;
	int *device_a=NULL, *device_b=NULL, *device_c=NULL;
	int **host_a = NULL, **host_b = NULL, **host_c = NULL;
	int size = this->N;
	int i, j;
	/*
		allocate main pointers
	*/
	cudaMallocHost((void **)&host_a, sizeof(int*)*size); // cpu-gpu memory bridge
	cudaMallocHost((void **)&host_b, sizeof(int*)*size);
	cudaMallocHost((void **)&host_c, sizeof(int*)*size);

	for (i = 0; i < size; i++)
	{
		/*
			2d array - allocate inner arrays
		*/
		cudaMallocHost((void **)&host_a[i], sizeof(int)*size);
		cudaMallocHost((void **)&host_b[i], sizeof(int)*size);
		cudaMallocHost((void **)&host_c[i], sizeof(int)*size);
	}


	/*
		copy data from this and other to host device pointer bridges
	*/
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			host_a[i][j] = this->arr[i][j];
			host_b[i][j] = other.arr[i][j];
			host_c[i][j] = matrix3.arr[i][j];
		}
	}




	cudaMalloc((void**) &device_a, sizeof(int)*size*size); // gpu memory allloc
	cudaMalloc((void**) &device_b, sizeof(int)*size*size);
	cudaMalloc((void**) &device_c, sizeof(int)*size*size);


	

	for (int i = 0; i < size; i++)
	{
		
		cudaMemcpy(device_a+i*size, host_a[i], sizeof(int)*size, cudaMemcpyHostToDevice); // copy mem to gpu
		cudaMemcpy(device_b+i*size, host_b[i], sizeof(int)*size, cudaMemcpyHostToDevice);
		cudaMemcpy(device_c+i*size, host_c[i], sizeof(int)*size, cudaMemcpyHostToDevice);
	}

		
	
	/* multi threadded mult call
	*/
	cuda_matrix_mult <<<size,size >>>(device_a,device_b,device_c,size); // function call
	
	cudaDeviceSynchronize();

	for (i = 0; i < size; i++)
	{
		cudaMemcpy(host_c[i], device_c+i*size, sizeof(int)*size, cudaMemcpyDeviceToHost); // copy results back

	}

	/*
		copy it to class
	*/
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			matrix3[i][j] = host_c[i][j];
		}
	}

	cudaThreadSynchronize();
	for (i = 0; i < size; i++)
	{
		/*
			free inner
		
		cudaFree(device_a[i]);
		cudaFree(device_b[i]);
		cudaFree(device_c[i]);
		*/
		cudaFreeHost(host_a[i]);
		cudaFreeHost(host_b[i]);
		cudaFreeHost(host_c[i]);
	}
	/*
		free-outer
	*/
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);

	return matrix3;
}
