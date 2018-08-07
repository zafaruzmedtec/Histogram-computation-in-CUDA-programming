#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <fstream>

using namespace std;

__global__ void histogram_compt(int *array , int *total_counter){
	
	int Thread_index = threadIdx.x;
	
	// global index
	int index = Thread_index + blockIdx.x * blockDim.x;

	// step 1: compute local/partial histogram (i.e. counter)
	
	__shared__ int partial_counter[8];
	
	
	if (Thread_index==0) { // only done by thread#0
		for(int i=0; i<8; i++){
			partial_counter[i] = 0; 
		}
	}
	__syncthreads();
	
	int Bin_index = array[index]/32;
	atomicAdd(&partial_counter[Bin_index], 1);
	
	
	// step 2: add partial histogram to global histogram

	if (Thread_index==0) {
		for(int i=0; i<8; i++){
			atomicAdd(&total_counter[i], partial_counter[i]);
			
		}
	}
	
	__syncthreads(); 
	
}	


const int imgH = 256;
const int imgW = 256;
const string filename = "/home/maia/cuda-workspace/Zafar/histogram1/img256.txt";

/* Read file and load to 2D array */
void loadFromFile(std::string filename, int mat[imgH][imgW]);

int main()
{	
	int array_length = imgH*imgW;
	int img[imgH][imgW]; //input
	loadFromFile(filename, img);
	//int array[] = img;
	
	const int TotalBins = 8; //bins
	int counter[TotalBins]; //counter
	
	//Device array
	int *dev_array, *dev_counter;
	
    //Allocate the memory on the GPU
    cudaMalloc((void **)&dev_array , array_length*sizeof(int) );
    cudaMalloc((void **)&dev_counter , TotalBins*sizeof(int) );
	
    //Copy Host array to Device array
    cudaMemcpy (dev_array , &img , array_length*sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy (dev_counter , &counter , TotalBins*sizeof(int) , cudaMemcpyHostToDevice);
    
	int threadsPerBlock = 256; //optimal
    //int blocksPerGrid =(array_length + threadsPerBlock - 1) / threadsPerBlock;
	float blocksPerGrid = array_length/threadsPerBlock;
	int blocks = floor(blocksPerGrid);
	
    cudaEvent_t start, stop;
	float time;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord (start, 0);
	
    //Make a call to GPU kernel, blocks, 256 threads
    histogram_compt <<< blocks, threadsPerBlock >>> (dev_array, dev_counter);
    
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