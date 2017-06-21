/**
 * @file   : readtxttofloatarray.cpp
 * @brief  : Read text file to C++ float array
 * @author : Ernest Yeung	ernestyalumni@gmail.com
 * @date   : 20170521
 * @ref    : cf. https://stackoverflow.com/questions/19102640/c-read-float-values-from-txt-and-put-them-into-an-unknown-size-2d-array
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
  * g++ -std=c++14 quicksort.cpp -o quicksort.exe
  * 
  * */

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <fstream>

int main() 
{
	std::fstream in("tweets2015stochasticgr_small.txt"); 
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
	
	std::cout << v[0][0] << v[0][1] << std::endl;
	std::cout << v.size() << std::endl;
	for (int i=0; i< v.size(); i++) {
		std::cout << v[i].size() << std::endl; }

/*
	for (int i=0; i< v.size(); i++) {
		std::cout << v[5][i] << " ";
		 }
*/ 


	std::cout << "Time to flatten this ... " << std::endl; 
	
	const int STRIDE = v[0].size();
	const int total_SIZE = v.size()*v.size();
	
	std::cout << v[STRIDE-1][STRIDE-1] << std::endl;

	
	std::vector<float> A_vec; 


	for (int i=0; i< v.size() ; i++) { 
		for (int j=0; j<STRIDE; j++) {
			A_vec.push_back( v[i][j] ); 
		}
	}
	
	
	std::cout << A_vec[5*STRIDE + 0] << " " << A_vec[5*STRIDE + 1] << " " << A_vec[5*STRIDE + 2] << std::endl; 
	
	std::cout << A_vec[(STRIDE-1)*STRIDE + STRIDE-3] << " " << A_vec[(STRIDE-1)*STRIDE + STRIDE-2] << " " << A_vec[(STRIDE-1)*STRIDE + STRIDE-1] << std::endl; 
	
	std::cout << A_vec.size() << std::endl; 

//	float A_flattened[STRIDE*v.size()] = A_vec.data(); 
	float* A_flattened = A_vec.data(); 

	std::cout << A_flattened[(STRIDE-1)*(STRIDE) + STRIDE-3] << " " << A_flattened[(STRIDE-1)*(STRIDE) + STRIDE-2] << std::endl;

}
