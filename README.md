# Computational Drug Discovery for SARS-COV-2 with Cov2GEN
*NOTE: This repository is created only for experimental purpose to see whether Graph Neural Net (GNN) is a good choice for drug discovery. **DO NOT TAKE definitive decision** based on these findings.*

**I'm an Electrical Engineer who happens to have a Masters Degree on Information Technology. I don't have any expert domain knowledge in the fields of Pharmacology or Molecular biology or Cheminformatics. However, I do have an immense interest to expand my knowledge on any one of these magnificent fields from a computational perspective. Before moving onto the following sections, I apologize in advance if I made any noob mistakes below. Thank you! Happy Reading!**

python version: 3.9.x

### Required Libraries/Packages
* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
* [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html)
* [chembl_webresource_client](https://github.com/chembl/chembl_webresource_client)
* [seaborn](https://seaborn.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)
* [pandas](https://pandas.pydata.org/)
* [tqdm](https://github.com/tqdm/tqdm)

### Read Me Instructions:
Following list of things will be explained:

0. [Explanation of the Core Idea](#0-explanation-of-the-core-idea)
1. [Experimental Setup](#1-experimental-setup)
2. [Design Cov2GEN model](#2-design-cov2gen-model)
3. [Experimental Results](#3-experimental-results)

# 0. Explanation of the Core Idea
* Every molecule or chemical compound or drugs composition can be considered as a Graph or Network of Atoms.
* Imagine a molecule, in which, each of the atoms is a connection point (nodes) and the chemical bonding between atoms is the connection (edges).
  With this in mind, this whole molecule can be considered as a network of atoms or a graph.
* When it comes to graphs, Graph Neural Nets are the new frontiners (frontiers??).
* In this little experiment, I attempted a GNN based approach to classify strong and weak inhibitors that can possibly act on a target protein of SARS Cov-2, in order to achieve desired effects.

# 1. Experimental Setup
### 1.1 Target Protein Selection and Bioactivity Data Download
* So let's get started. For this little experiment, I've chosen a target protein of Coronavirus.
  But for SARS-Cov2, the only target single protein I have found is *Replicase polyprotein 1ab*.
* Now for that target protein I have downloaded bioactivity data. I have considered those compounds that have IC50 values.
  By now, you might have got the idea, that my ultimate target is to find the inhibiting compounds / drugs / molecules that are likely to have strong potency.
* From the IC50 values, I have classfied these molecules into two classes. The molecules with IC50 values less than 1000 nanoMolar (1uM) have been considered to be strong inhibitors and the ones greater than 1uM to be weak inhibitors.

### 1.2 Build PyG (Pytorch Geometric) Graph Data
* This step takes quite a bit of journey.
* Firstly, all of the atoms inside molecules have been considered as nodes and the connection between them have been considered as edges.
  Then, with the help of RDKit, all of the atoms and bonds between them have been featurized.
* Next, the model input ready PyG graph data has been processed.
* I have found only 102 compounds that can act on the desired target protein. Dataset Splits and Configurations are as follows:
  |Total|Train|Test|
  |----|----|----|
  |102|76|26|

# 2. Design Cov2GEN model
* This is a very simple and straightforward Graph Neural Net Architecture. I have named it **Cov2GEN**.
* At the core of Cov2GEN, Long short-term memory (LSTM) and GENeralized Graph Convolution (GENConv) have been used.
* Look at the following picture to get a high level overview.

![image](https://github.com/AsifAlFaisal/cov2-drug-discovery-demo/blob/main/saved_model/output_images/Cov2GEN_Arch.png) <br/>

# 3. Experimental Results
* For a classification problem, relying only on **Accuracy** sometimes cause accuracy paradox.
  Therefore, for more credibility, other metrics like **Precision Score**, **Recall Score** and **F1-Score** can be used to understand true performance of the model.
  * **Precision** is a metric that answers the question, *What proportion of positive identifications is actually correct?*. 
  * **Recall** tries to answer the question, *What proportion of the actual positives is identified correctly?*
  * **F1 Score** is the harmonic mean of precision and recall, which indicates the balance between these two.
* The test result on trained model comes out as follows:
  |Precision|Recall|F1-Score|Accuracy|Loss|
  |----|----|----|----|----|
  |0.5556|0.8333|0.6667|0.6154|0.1035|

Metric scores for the test data are fairly good, considering I have very little amount of data to train and test. It kinda indicates that the model may generalize quite well over the large amount of data.
- And from the figure below, we see that in the test data, out of 12 strong inhibitors, 10 of them have been accurately classified, which seems to be quite good, considering the volumn of training and test data.

![image](https://github.com/AsifAlFaisal/cov2-drug-discovery-demo/blob/main/saved_model/output_images/confusion_matrix_test.png) <br/>