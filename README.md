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



## Original version, `./original/`
### Big example  

We had successfully parsed, preprocessed (data cleaned), and applied TextRank (`nltk`'s "`pagerank`" (Python) implementation) on a portion of this massive (nearly 2000 pages) reference manual:

http://www.ti.com/lit/ug/spnu562/spnu562.pdf   

### `mypolly.ai` chat bot Front End

![mypolly.ai webpage API](https://github.com/ernestyalumni/txtpdfminerBot/raw/master/images/mypollyScreenshot%20from%202017-06-03%2019-03-41.png) 

![Mayo, Facebook Messenger chatbot interface](https://github.com/ernestyalumni/txtpdfminerBot/raw/master/images/MayoScreenshot%20from%202017-06-03%2019-02-47.png)




