## Computational Drug Discovery for SARS-COV-2 using Cov2GEN
*NOTE: This repository is created only for experimental purpose to see whether Graph Neural Net (GNN) is a good choice for drug discovery. DO NOT TAKE definitive decision based on these findings.*

python version: 3.9.x

### Required Libraries/Packages
- PyTorch
- PyTorch Geometric
- rdKit
- chembl_webresource_client
- seaborn
- scikit-learn
- pandas
- tqdm

### Read Me Instructions:

Following list of things will be explained:
0. Core Idea Explanation
1. Raw Data Download and Preprocessing
2. Build PyG (Pytorch Geometric) Graph Data
3. Design Cov2GEN model
4. Experimental Results


# 4. Experimental Results

- For a classification problem, relying only on **Accuracy** sometimes may cause accuracy paradox.
<br/>So, for more credibility, other metrics like **Precision Score**, **Recall Score** and **F1-Score** can be used to understand true performance of the model.
<br/>**Precision** is a metric that answers the question, *What proportion of positive identifications is actually correct?*. 
<br/>**Recall** tries to answer the question, *What proportion of the actual positives is identified correctly?*
<br/>**F1 Score** is the harmonic mean of precision and recall, which indicates the balance between these two.

- I have calculated different metrics to check model performance. 

- The test result on trained model comes out as follows:

|Precision|Recall|F1-Score|Accuracy|Loss|
|----|----|----|----|----|
|0.5556|0.8333|0.6667|0.6154|0.1035|

Metric scores for the test data are fairly good, considering I have very little amount of data to train and test. It kinda indicates that the model may generalized quite well over the large amount of data.
- And from the below figure we can see that out 12 strong inhibitors 10 of them are accurately classified, which seems to be quite good regardless the volumn of training and test data.

![image](https://github.com/AsifAlFaisal/cov2-drug-discovery-demo/blob/main/saved_model/output_images/confusion_matrix_test.png) <br/>
