/**
 * @file   : pagerank.h
 * @brief  : file for PageRank implementation
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170620
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
#ifndef __PAGERANK_H__
#define __PAGERANK_H__

#include <iostream> 
#include "cublas_v2.h" 

/**
 * @brief : pageRank_dense is pageRank, but for dense matrices 
 * INPUT(S)/ARGUMENTS
 * float* B := stochastic matrix; we assume we're given a stochastic matrix that 
 * satisfies the condition that the sum of entries along a single column adds up to 1
 * Note that B is in column-major format, since CUBLAS is in column-major format
 * */
void pageRank_dense(const float alpha, const float EPS, 
						const int N,
						float* B, float* x, float* y, float* E,
						float* diffyx, const unsigned int maxIters) ;

#endif // __PAGERANK_H__
