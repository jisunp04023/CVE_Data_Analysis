# CVE_Data_Analysis

1. DOWNLOAD URL: 'https://github.com/jisunp04023/CVE_Data_Analysis.git'


2. INSTALL REQUIRES:
  * Python 3.6
  * numpy >= 1.12.0
  * scipy >= 0.19.0
  * networkx == 2.4
  * scikit-learn >= 0.21.2
  * theano >= 0.9.0
  * keras == 2.0.2


3. EXECUTION
  * Locate input data(=cve_list.txt) in execute/input.
  * To analyze the data, from the execute folder:
    - first, run 'python DataAnalysis.py'
    - second, run 'python GraphSummarization.py'

  * Check the results from the results folder:
    - DataAnalysis.xlsx : analaysis of the CVE data
    - Embedding.png     : visualization of the embedded graph using HOPE
    - Clustering.png    : visualization of the clustered output using K-means clustering
    - ClusterTable.csv  : information of which cluster each CVE is allocated in
