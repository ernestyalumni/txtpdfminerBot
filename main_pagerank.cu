/**
 * @file   : main_pagerank.cu
 * @brief  : main file for PageRank implementation
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170619
 * @ref    : 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * P.S. I'm using an EVGA GeForce GTX 980 Ti which has at most compute capability 5.2; 
 * A hardware donation or financial contribution would help with obtaining a 1080 Ti with compute capability 6.2
 * so that I can do -arch='sm_62' and use many new CUDA Unified Memory features
 * */
 /**
  * COMPILATION TIP(s)
  * (without make file, stand-alone)
  * nvcc -std=c++11 -arch='sm_61' main_pagerank.cu -o main_pagerank.exe
  * 
  * */
#include <iostream> 

#include <vector>
//#include <array>
#include <string>
#include <sstream>
#include <fstream>

#include "pagerank.h"


// these have to be set manually 
constexpr const int N {2018};
constexpr const int Nsquared {2018*2018};

constexpr const int N_eg { 9 };
constexpr const int Nsquared_eg {81};

// stochastic matrix
__device__ __managed__ float B[Nsquared];  

__device__ __managed__ float B_eg[Nsquared_eg];  


// PageRank values for each vertex, as a vector (array)
__device__ __managed__ float x[N] ; 
__device__ __managed__ float y[N] ; 

__device__ __managed__ float E[N] ;

__device__ __managed__ float diffyx[N]; 


__device__ __managed__ float x_eg[N_eg] ; 
__device__ __managed__ float y_eg[N_eg] ; 

__device__ __managed__ float E_eg[N_eg] ;

__device__ __managed__ float diffyx_eg[N_eg]; 



std::vector<std::vector<float>> read_array_txt(char* filename) 
{
		// std::fstream in("tweets2015stochasticgr_small.txt"); 
	std::fstream in(filename); 
	std::string line;
	std::vector<std::vector<float>> v;
	int i =0;
	while (std::getline(in, line)) 
	{
		float value;
		std::stringstream ss(line);
		
		v.push_back(std::vector<float>()); 
		
		while (ss >> value)
		{
			v[i].push_back(value);
		}
		++i;
	}
	return v; 
}

// Note, this makes a COLUMN-MAJOR flattened array, in accordance with CUBLAS's format 
std::vector<float> make_flattened_vec_array( std::vector<std::vector<float>> & v, const int STRIDE) 
{
	std::vector<float> A_vec; 

	for (int j=0; j<STRIDE; j++) {
		for (int i=0; i< v.size() ; i++) { 
			A_vec.push_back( v[i][j] ); 
		}
	}

	return A_vec;
}

// Note, this makes a ROW-MAJOR flattened array, in accordance with CUBLAS's format 
std::vector<float> make_flattened_vec_array_ROWMAJOR( std::vector<std::vector<float>> & v, const int STRIDE) 
{
	std::vector<float> A_vec; 

	for (int i=0;i<v.size(); i++) { 
		for (int j=0; j<STRIDE; j++) {
			A_vec.push_back( v[i][j] ); 
		}
	}

	return A_vec;
}


int main(int argc, char* argv[]) 
{
	// initiate example B_eg (small B to try first)
	// B_eg is in COLUMN-MAJOR format
	for (int j=0;j<N_eg;j++) {
		for (int i=0;i<N_eg;i++) {
			B_eg[i + j *N_eg] = 0.f; }
	}
	B_eg[0 + 4 *N_eg ] = 1.0f;
	B_eg[1 + 6 *N_eg ] = 0.5f;
	B_eg[2 + 7 *N_eg ] = 1.0f;
	B_eg[3 + 1 *N_eg ] = 0.333333f;
	B_eg[4 + 0 *N_eg ] = 1.f;
	B_eg[4 + 1 *N_eg ] = 0.333333f;
	B_eg[4 + 2 *N_eg ] = 1.0f;
	B_eg[4 + 3 *N_eg ] = 1.0f;
	B_eg[5 + 6 *N_eg ] = 0.5f;
	B_eg[6 + 1 *N_eg ] = 0.333333f;
	B_eg[7 + 5 *N_eg ] = 1.0f;
	B_eg[7 + 8 *N_eg ] = 1.0f;

	cudaDeviceSynchronize(); 
	for (int i=0;i<N_eg;i++) {
		for (int j=0;j<N_eg;j++) {
			std::cout << B_eg[i + j *N_eg] << " "; }
		std::cout << std::endl; 
	}

	cudaMemset(y_eg,0.f,N_eg*sizeof(float));
//	cudaMemset(x_eg,1.f/N_eg,N_eg*sizeof(float));
	cudaDeviceSynchronize();
	std::cout << " x_eg, before multiplication " << std::endl; 
	for (int i=0; i < N_eg; ++i) {
		x_eg[i] = 1.f/((float) N_eg);
		std::cout << x_eg[i] << " ";  } 
	std::cout << std::endl ;	
	for (int i=0; i < N_eg; ++i) {
		y_eg[i] = 0.f;
		std::cout << y_eg[i] << " ";  } 
	std::cout << std::endl ;	

	// Set E_eg
	for (int i=0; i<N_eg; ++i) {
		E_eg[i] = 1.0f/N_eg;
	}

	float a1=1.0f;		// a1=1
	float bet=0.0f; 	// bet = 0

	float a2 = 0.85;
	float a3 = (1.0f - a2); 

	float a4 = -1.0f;  
	cublasHandle_t handle_eg;	// CUBLAS context
	cublasCreate(&handle_eg);
	
	cublasSgemv(handle_eg,CUBLAS_OP_N,N_eg,N_eg,&a1,B_eg,N_eg,x_eg,1,&bet,y_eg,1);
	cudaDeviceSynchronize();


	std::cout << " y_eg , after matrix multiplication: " << std::endl; 
	for (int i=0; i < N_eg; ++i) {
		std::cout << y_eg[i] << " ";  } 
	std::cout << std::endl ;	
	
	cublasSscal(handle_eg,N_eg,&a2,y_eg,1); 
	cudaDeviceSynchronize();  
		
	// add (1.0f - alpha) * E, i.e.
	// y = (1.0f - alpha) * E + y
	cublasSaxpy(handle_eg,N_eg,&a3,E_eg,1,y_eg,1);
	cudaDeviceSynchronize(); 

	std::cout << " y_eg , after scaling and vector addition: " << std::endl; 
	for (int i=0; i < N_eg; ++i) {
		std::cout << y_eg[i] << " ";  } 
	std::cout << std::endl ;	

	float err_eg = 0.0f;

	cublasSaxpy(handle_eg,N_eg,&a1,y_eg,1,diffyx_eg,1); 
	cudaDeviceSynchronize();
	cublasSaxpy(handle_eg,N_eg,&a4,x_eg,1,diffyx_eg,1);
	cudaDeviceSynchronize(); 
	cublasSasum(handle_eg,N_eg,diffyx_eg,1,&err_eg); 
	cudaDeviceSynchronize(); 

	std::cout << " diffyx_eg : " << std::endl; 
	for (int i=0; i < N_eg; ++i) {
		std::cout << diffyx_eg[i] << " ";  } 
	std::cout << std::endl ;	

	float sum_eg = 0.f;
	std::cout << " err_eg : " << err_eg << std::endl; 
	cublasSasum(handle_eg,N_eg,x_eg,1,&sum_eg); 
	cudaDeviceSynchronize(); 
	std::cout << " sum_eg : " << sum_eg << std::endl; 
	
	pageRank_dense(a2, 0.0001f, N_eg,B_eg,x_eg,y_eg,E_eg,diffyx_eg,2) ;

	std::cout << " x_eg , after pagerank: " << std::endl; 
	for (int i=0; i < N_eg; ++i) {
		std::cout << x_eg[i] << " ";  } 
	std::cout << std::endl ;	

	pageRank_dense(a2, 0.0001f, N_eg,B_eg,x_eg,y_eg,E_eg,diffyx_eg,3) ;

	std::cout << " x_eg , after 3 iterations of pagerank: " << std::endl; 
	for (int i=0; i < N_eg; ++i) {
		std::cout << x_eg[i] << " ";  } 
	std::cout << std::endl << std::endl;	

	pageRank_dense(a2, 0.0001f, N_eg,B_eg,x_eg,y_eg,E_eg,diffyx_eg,4) ;

	std::cout << " x_eg , after 4 iterations of pagerank: " << std::endl; 
	for (int i=0; i < N_eg; ++i) {
		std::cout << x_eg[i] << " ";  } 
	std::cout << std::endl << std::endl;	

	
	// END of small example
	

	// Set E
	for (int i=0; i<N; ++i) {
		E[i] = 1.0f/N;
	}



	char inputfilename[] = "tweets2015stochasticgr_small.txt"; 
	std::vector<std::vector<float>> B_vec = read_array_txt(inputfilename); 
		
	// Note, this makes a COLUMN-MAJOR flattened array, in accordance with CUBLAS's format 
//	std::vector<float> B_flattened = make_flattened_vec_array(B_vec,N); 	

	// Note, this makes a ROW-MAJOR flattened array, in accordance with CUBLAS's format 
	std::vector<float> B_flattened = make_flattened_vec_array_ROWMAJOR(B_vec,N); 	


	// initialize B
	for (int k =0; k< Nsquared; k++) { 
		B[k] = B_flattened[k]; }

	// sanity check
	std::cout << "\n \n B : " << std::endl; 
	for (int i=0; i < 10; ++i) {
		std::cout << B[i] << " ";  } 
	std::cout << std::endl ;	
	for (int i=Nsquared-12; i < Nsquared; ++i) {
		std::cout << B[i] << " ";  } 
	std::cout << std::endl ;	
	float sanitysum0 = 0.f; // column 0 summation
	for (int i=0;i<N;i++) {
		sanitysum0 += B[i+0*N]; }
	std::cout << "\n Sanity check of summation of column 0 : " << sanitysum0 << std::endl;
	float sanitysum1 = 0.f; // column 0 summation
	for (int i=0;i<N;i++) {
		sanitysum1 += B[i+1*N]; }
	std::cout << "\n Sanity check of summation of column 1 : " << sanitysum1 << std::endl;
	float sanitysum2 = 0.f; // column 0 summation
	for (int i=0;i<N;i++) {
		sanitysum2 += B[i+2*N]; }
	std::cout << "\n Sanity check of summation of column 2 : " << sanitysum2 << std::endl;
	float sanitysum3 = 0.f; // column 0 summation
	for (int i=0;i<N;i++) {
		sanitysum3 += B[i+3*N]; }
	std::cout << "\n Sanity check of summation of column 3 : " << sanitysum3 << std::endl;
	float sanityrowsum0 = 0.f; // row 0 summation
	for (int j=0;j<N;j++) {
		sanityrowsum0 += B[0+j*N]; }
	std::cout << "\n Sanity check of summation of row 0 : " << sanityrowsum0 << std::endl;
	float sanityrowsum1 = 0.f; // row 0 summation
	for (int j=0;j<N;j++) {
		sanityrowsum1 += B[1+j*N]; }
	std::cout << "\n Sanity check of summation of row 1 : " << sanityrowsum1 << std::endl;
	float sanityrowsum2 = 0.f; // row 0 summation
	for (int j=0;j<N;j++) {
		sanityrowsum2 += B[2+j*N]; }
	std::cout << "\n Sanity check of summation of row 2 : " << sanityrowsum2 << std::endl;
	float sanityrowsum3 = 0.f; // row 0 summation
	for (int j=0;j<N;j++) {
		sanityrowsum3 += B[3+j*N]; }
	std::cout << "\n Sanity check of summation of row 3 : " << sanityrowsum3 << std::endl;
	
	


	pageRank_dense(0.99999f, 1.e-10f, N,B,x,y,E,diffyx,100); 


	// sanity check
	std::cout << " x : " << std::endl; 
	for (int i=0; i < 30; ++i) {
		std::cout << x[i] << " ";  } 
	std::cout << std::endl ;	

	
	
	cublasHandle_t handle;	// CUBLAS context
	cublasCreate(&handle);

	int maxresult;
	cublasIsamax(handle,N,x,1,&maxresult);
	
	int minresult;
	cublasIsamin(handle,N,x,1,&minresult);

	std::cout << "i for max is  : " << maxresult << std::endl;
	std::cout << "i for min is  : " << minresult << std::endl;
	
	std::cout << "max |x[i]| is : " << fabs(x[maxresult-1]) << std::endl;
	std::cout << "min |x[i]| is : " << fabs(x[minresult-1]) << std::endl;


	// C++ File I/O to write result of PageRank  
	std::ofstream out_file;  
	
	out_file.open("Pagerankresult.txt");
	for (int i=0;i<N;++i) {
		out_file << x[i] << " ";
	}
	out_file << std::endl;
	out_file.close();   
	
	

	cublasDestroy( handle_eg );
}
