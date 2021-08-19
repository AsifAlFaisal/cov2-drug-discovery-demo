# Computational Drug Discovery for SARS-COV-2 with Cov2GEN
*NOTE: This repository is created only for experimental purpose to see whether Graph Neural Net (GNN) is a good choice for drug discovery. **DO NOT TAKE definitive decision** based on these findings.*

**I'm an Electrical Engineer who happens to have a Masters Degree on Information Technology. I don't have expert domain knowledge in the fields of Pharmacology or Molecular biology or Cheminformatics. However, I do have immense interest to expand my knowledge on any one of these magnificent fields from a computational perspective. Before moving onto the following sections, I apologize in advance if I made any noob mistakes below. Thank you! Happy Reading!**

python version: 3.9.x

### Required Libraries/Packages
- PyTorch
- PyTorch Geometric
- RDKit
- chembl_webresource_client
- seaborn
- scikit-learn
- pandas
- tqdm

### Read Me Instructions:

Following list of things will be explained:

0. Core Idea Explanation
1. Experimental Setup
2. Design Cov2GEN model
3. Experimental Results

# 0. Core Idea Explanation

- Every molecules or chemical compounds or drugs composition can be considered as Graphs or Networks of Atoms.

- Think of a molecule where each atom could be a connection point (nodes) and the chemical bonding between atoms could be the connection (edges).
<br/>With that idea in mind, a whole molecule can be considered as a network of atoms or a graph.

- When it comes to graphs, Graph Neural Nets are the new frontiners.

- So in this little experiment, I attempted a GNN based approach to classify strong and weak inhibitors that possibly can act on a target protein of SARS Cov-2, in order to achieve desired effects.

# 1. Experimental Setup

### 1.1 Target Protein Selection and Bioactivity Data Download

- So, here we are. For this little experiment I've chosen a target protein of Coronavirus.
<br/>But for SARS-Cov2, the only target single protein I have found in ChEMBL is *Replicase polyprotein 1ab*.

- Now for that target protein I have downloaded bioactivity data. I have considered those compounds that have IC50 values.
<br/>So now you probably get the idea that my ultimate target is to find inhibiting compounds/drugs/molecules that will have strong potency.

- From the IC50 values, I have classfied these molecules into two classes. IC50 values less than 1000 nanoMolar (1uM) are considered strong inhibitors and greater than 1uM considered weak inhibitors.

### 1.2 Build PyG (Pytorch Geometric) Graph Data

- This step takes quite a bit of journey. 

- First all the atoms inside molecules considered as nodes and the connection between them considered as edges.
<br/>Then with the help of RDKit all the atoms and bonds between them have been featurized.

- After that, PyG graph data for model input has been processed.

- I have found only 102 compounds that can act on the desired target protein. Dataset Splits and Configuration are as follows:

|Total|Train|Test|
|----|----|----|
|102|76|26|

# 2. Design Cov2GEN model

- This is a very simple and straightforward Graph Neural Net Architecture. I have named it as **Cov2GEN**.

- At the core of Cov2GEN, Long short-term memory (LSTM) and GENeralized Graph Convolution (GENConv) are used.

- Look at the following picture to get the high level overview.

![image](https://github.com/AsifAlFaisal/cov2-drug-discovery-demo/blob/main/saved_model/output_images/Cov2GEN_Arch.png) <br/>

# 3. Experimental Results

- For a classification problem, relying only on **Accuracy** sometimes may cause accuracy paradox.
<br/>So, for more credibility, other metrics like **Precision Score**, **Recall Score** and **F1-Score** can be used to understand true performance of the model.
<br/>**Precision** is a metric that answers the question, *What proportion of positive identifications is actually correct?*. 
<br/>**Recall** tries to answer the question, *What proportion of the actual positives is identified correctly?*
<br/>**F1 Score** is the harmonic mean of precision and recall, which indicates the balance between these two.

- The test result on trained model comes out as follows:

|Precision|Recall|F1-Score|Accuracy|Loss|
|----|----|----|----|----|
|0.5556|0.8333|0.6667|0.6154|0.1035|

Metric scores for the test data are fairly good, considering I have very little amount of data to train and test. It kinda indicates that the model may generalized quite well over the large amount of data.
- And from the below figure we can see that in test data, out of 12 strong inhibitors 10 of them are accurately classified, which seems to be quite good considering the volumn of training and test data.

![image](https://github.com/AsifAlFaisal/cov2-drug-discovery-demo/blob/main/saved_model/output_images/confusion_matrix_test.png) <br/>
