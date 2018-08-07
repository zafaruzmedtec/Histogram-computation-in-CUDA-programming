#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <fstream>

using namespace std;
const int imgH = 256;
const int imgW = 256;
const string filename = "/home/maia/cuda-workspace/Zafar/histogram1/img256.txt";

/* Read file and load to 2D array */
void loadFromFile(std::string filename, int mat[imgH][imgW]);

int main()
{

	int img[imgH][imgW];

	loadFromFile(filename, img);

	const int TotalBins = 8;
	int counter[TotalBins];

	clock_t t;
	t = clock();

	for(int i=0; i<TotalBins; i++){
		counter[i]=0; //initializing bins with 0
	}

	for(int i=0; i<imgH; i++){
		for(int j=0; j<imgW; j++){
			int binIdx = (int)img[i][j]/32;
			counter[binIdx]++;
		}
	}

	t = clock() - t;
	cout << "Ellapsed Time: " << t << " miliseconds" << endl;
	cout << CLOCKS_PER_SEC << " clocks per second" << endl;
	cout << "Ellapsed Time: " << t*1.0/CLOCKS_PER_SEC << " seconds" << endl;


	for(int i=0; i<TotalBins; i++){
		cout<<"Bins "<<i<<" :"<<counter[i]<<endl;
	}
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

