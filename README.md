# txtpdfminerBot
*Bot that mines txt and pdfs*

**Authors**:    
*Back-end*  - Ernest Yeung [ernestyalumni at gmail dot com], [@ernestyalumni](https://twitter.com/ernestyalumni)    
*Front-end* - Kuhan Muniam     

## Features (updated June 21, 2017)
- CUDA C/C++ implementation of PageRank for *dense* stochastic matrices, using
  * `CUBLAS`
  * CUDA Unified Memory Management   
  resulting in a scalable (multi-GPU) [PageRank algorithm](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) implementation   

![CUDA C/C++ executable `main_pagerank.exe` for PageRank with dense matrices, CUBLAS](https://github.com/ernestyalumni/txtpdfminerBot/raw/master/images/CUDACppPageRankCUBLASScreenshot%20from%202017-06-21%2015-16-14.png)

- PageRank (TextRank) of tweets (in 2015) on twitter from [`ntlk`](http://www.nltk.org/) corpus dataset

![examples of tweets from ntlk corpus](https://github.com/ernestyalumni/txtpdfminerBot/raw/master/images/tweetexamples_ntlk_graphs_metrics_Screenshot%20from%202017-06-21%2015-09-41.png)   
    
------------------------------------------------------------->    

![Top 75 words](https://github.com/ernestyalumni/txtpdfminerBot/raw/master/images/Top75wordstweets2015Screenshot%20from%202017-06-21%2015-11-48.png)

## (Original) features (cf. `./original/`)   

- Parsed pdf and preprocessed ((data) cleaned/wrangled) data with Python `pdfminer`   
- implemented TextRank by constructing a graph with Python library `networkx` and Python library `nltk` and its PageRank implementation
- results are exported in `json` format to populate `json` fields for the myPolly.ai chat bot, named "Mayo."  Facebook Messenger interface was easily deployed with myPolly.ai browser API/interface.  

## June 3rd, 2017 update; we won a prize for this chat bot at the [MyPolly.ai Botathon](https://www.meetup.com/AI-LA-Meetup/events/239292598/) in Los Angeles!  Thank you to the organizer, judges, and datalog.ai!  

We won a prize in recognition of the technical process involved with the backend, in implementing unsupervised learning on a large corpus of text with the PageRank/TextRank algorithm.  

## Updated backend, in `./`

- [`./pagerank.cu`](https://github.com/ernestyalumni/txtpdfminerBot/raw/master/pagerank.cu)
  * Numerical Computation of large *dense* matrices of matrix size dimensions * 2000 x 2000* is computational expensive in Python; `nltk`'s pagerank implementation froze up when I tried it on a portion of the 2015 tweets from the `ntlk` Twitter corpus.  Matrix operations with large dense matrices are not a problem with CUDA C/C++, in particular, its `CUBLAS` library.    
  * Along with `CUBLAS`, implementation with *CUDA Unified Memory Management* makes this implementation scalable (*multiGPU supported*).     

- [`./pagerank.h`](https://github.com/ernestyalumni/txtpdfminerBot/raw/master/pagerank.h)   
  * CUDA C/C++ header file for `./pagerank.cu`

- [`./main_pagerank.cu`](https://github.com/ernestyalumni/txtpdfminerBot/raw/master/main_pagerank.cu)    
  * "main file" that
    1.  does *File I/O*, importing the text file with the matrix for use as a **COLUMN-MAJOR** matrix format and  
    2.  runs the PageRank CUDA C++ function for dense matrices

The 3 files above are the "meat" of the PageRank implementation in CUDA C/C++/CUBLAS.  

- [`./ntlk_graphs_metrics.ipynb`](https://github.com/ernestyalumni/txtpdfminerBot/blob/master/ntlk_graphs_metrics.ipynb)
  * jupyter notebook (Python) that steps through the construction of the graph for the example of `nltk`'s corpus for Twitter.    
    

## Original version, `./original/`
### Big example  

First, we showed an implementation of the TextRank algorithm for finding relevant keywords in a text (i.e. unsupervised learning) for a simple example, [`10.txt`](https://github.com/ernestyalumni/txtpdfminerBot/blob/master/original/10.txt)

Then, we had successfully parsed, preprocessed (data cleaned), and applied TextRank (`nltk`'s "`pagerank`" (Python) implementation) on a portion of this massive (nearly 2000 pages) reference manual:

http://www.ti.com/lit/ug/spnu562/spnu562.pdf   

This was shown, step-by-step, in jupyter notebook (Python) [`TextSummarization_edited.ipynb`](https://github.com/ernestyalumni/txtpdfminerBot/blob/master/original/TextSummarization_edited.ipynb).  

Note that for *graph construction* the **Levenshtein** metric between 2 strings was used.   

### `mypolly.ai` chat bot Front End

![mypolly.ai webpage API](https://github.com/ernestyalumni/txtpdfminerBot/raw/master/images/mypollyScreenshot%20from%202017-06-03%2019-03-41.png) 

![Mayo, Facebook Messenger chatbot interface](https://github.com/ernestyalumni/txtpdfminerBot/raw/master/images/MayoScreenshot%20from%202017-06-03%2019-02-47.png)

## References

I dive into the theory/mathematics behind the PageRank algorithm in the section for "Unsupervised Learning" in my notes on [Machine Learning](https://github.com/ernestyalumni/MLgrabbag/raw/mobile/LaTeXandpdfs/ML.pdf)


