/**
 * @file   : pagerank.cu
 * @brief  : file for PageRank implementation
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
#include "pagerank.h"

void pageRank_dense(const float alpha, const float EPS, 
						const int N, // number of vertices or webpages or tokens
						float* B, float* x, float* y, float* E,
						float* diffyx, const unsigned int maxIters) 
{
	// CUBLAS
	cublasStatus_t stat; 	// CUBLAS functions status
	cublasHandle_t handle;	// CUBLAS context
	cublasCreate(&handle);

	float a1=1.0f;		// a1=1
	float bet=0.0f; 	// bet = 0

	float a2 = alpha;
	float a3 = (1.0f - alpha); 

	float a4 = -1.0f;  

	/************************/
	/* prepare x      		*/
	/************************/
	// initialize x with 1/N, where N=(total) number of vertices (webpages or words)
	// initialize y with 0 
	for (int i =0; i<N; ++i) {
		x[i] = 1.0f/((float) N);
		y[i] = 0.0f;
	}
	
	/************************/
	/* find eigenvalue      */
	/* (loop)  				*/
	/************************/
	float err = 2.0f * EPS; // choose something bigger than EPS initially

	int iteration = 0;

	float sum = 0.0f;
	while( (iteration < maxIters) && (err > EPS)) 
	{
		err = 0.0f;
		cudaMemset(diffyx,0.f,N*sizeof(float));
		
		// matrix-vector multiplication: y = B*x
		stat=cublasSgemv(handle,CUBLAS_OP_N,N,N,&a1,B,N,x,1,&bet,y,1);		
		cudaDeviceSynchronize();

		// do regularization
		// scale the vector y
		 
		cublasSscal(handle,N,&a2,y,1); 
		cudaDeviceSynchronize();  
		
		// add (1.0f - alpha) * E, i.e.
		// y = (1.0f - alpha) * E + y
		cublasSaxpy(handle,N,&a3,E,1,y,1);
		cudaDeviceSynchronize(); 
				
		// calculate error
		stat=cublasSaxpy(handle,N,&a1,y,1,diffyx,1); 
		cudaDeviceSynchronize();
		stat=cublasSaxpy(handle,N,&a4,x,1,diffyx,1);
		cudaDeviceSynchronize(); 
		stat=cublasSasum(handle,N,diffyx,1,&err); 
		cudaDeviceSynchronize(); 

		// this function copies the vector y into the vector x
		stat=cublasScopy(handle,N,y,1,x,1);
		cudaDeviceSynchronize(); 

		cudaMemset(y,0.f,N*sizeof(float));

		// sanity check
//		std::cout << " err : " << err << std::endl; 	


		iteration++; 	
	}
	stat=cublasSasum(handle,N,x,1,&sum); 
	cudaDeviceSynchronize(); 

	std::cout << " Iterations : " << iteration << std::endl; 
	std::cout << " err : " << err << std::endl; 	
	std::cout << " sum : " << sum << std::endl; 
	
	cublasDestroy( handle );
}




