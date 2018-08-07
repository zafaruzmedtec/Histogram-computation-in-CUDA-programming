#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <fstream>

using namespace std;

const string filename = "/home/maia/cuda-workspace/Zafar/histogram1/img256.txt";
const int imgH = 256;
const int imgW = 256;
int array_length = imgH*imgW;

int threadsPerBlock = 256; //optimal
int pixels_per_thread = 16;
float blocksPerGrid = array_length/threadsPerBlock; //256*256/256=256
int blocks = floor(blocksPerGrid)/pixels_per_thread; //256/16=16
int total_threads = blocks*threadsPerBlock; //16*256=4096


__global__ void histogram_compt(int *array , int *total_counter, int pixels_per_thread, int total_threads){

	// global index
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	// step 1: compute local/partial histogram (i.e. counter)

	int partial_counter[8];

	for(int i=0; i<8; i++){
		partial_counter[i] = 0;
	}

	for(int i=0; i<pixels_per_thread; i++){
		int Bin_index = array[index + (total_threads * i)]/32;
		partial_counter[Bin_index]++;
	}

	// step 2: add partial histogram to global histogram

	for(int i=0; i<8; i++){
		atomicAdd(&total_counter[i], partial_counter[i]);
	}

}




/* Read file and load to 2D array */
void loadFromFile(std::string filename, int mat[imgH][imgW]);

int main()
{
	//int *pixels_per_thread_pointer = &pixels_per_thread;
	//int *total_threads_pointer = &total_threads;
	int img[imgH][imgW]; //input
	loadFromFile(filename, img);

	const int TotalBins = 8; //bins
	int counter[TotalBins]; //counter

	//Device array
	int *dev_array, *dev_counter, //*dev_pixels_per_thread, *dev_total_threads;

    //Allocate the memory on the GPU
    cudaMalloc((void **)&dev_array , array_length*sizeof(int) );
    cudaMalloc((void **)&dev_counter , TotalBins*sizeof(int) );
    //cudaMalloc((void **)&dev_pixels_per_thread , pixels_per_thread_pointer*sizeof(int) );
    //cudaMalloc((void **)&dev_total_threads , total_threads_pointer*sizeof(int) );


    //Copy Host array to Device array
    cudaMemcpy (dev_array , &img , array_length*sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy (dev_counter , &counter , TotalBins*sizeof(int) , cudaMemcpyHostToDevice);
    //cudaMemcpy (dev_pixels_per_thread , &pixels_per_thread_pointer , pixels_per_thread_pointer*sizeof(int) , cudaMemcpyHostToDevice);
    //cudaMemcpy (dev_total_threads , &total_threads_pointer , total_threads_pointer*sizeof(int) , cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
	float time;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord (start, 0);

    //Make a call to GPU kernel, blocks, 256 threads
    histogram_compt <<< blocks, threadsPerBlock >>> (dev_array, dev_counter, pixels_per_thread, total_threads); //*dev_pixels_per_thread, *dev_total_threads

	cudaEventRecord (stop, 0);
	cudaEventSynchronize (stop);
	cudaEventElapsedTime (&time, start, stop);
	cudaEventDestroy (start);
	cudaEventDestroy (stop);


    //Copy back to Host array from Device array
    cudaMemcpy (&counter , dev_counter , TotalBins*sizeof(int) , cudaMemcpyDeviceToHost);

    for(int i=0; i<TotalBins; i++){
		cout<<"Bins "<<i<<" :"<<counter[i]<<endl;
	}

    cout << "Ellapsed Time: " << time << endl;

    cudaFree(dev_array);
    cudaFree(dev_counter);

    return 0;

}

void loadFromFile(string filename, int mat[imgH][imgW]) {
	std::ifstream fin;
	fin.open(filename.c_str());
	if (!fin) { std::cerr << "cannot open file"; }
	for (int i = 0; i<imgH; i++) {
		for (int j = 0; j<imgW; j++) {
			fin >> mat[i][j];
		}
	}
	fin.close();
}
