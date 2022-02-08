# vec2vec

This repository provides a reference implementation of vec2vec, which can reduce the dimension to matrix.

For the algorithm of vec2vec, you can read the following paper:

```
Wang X, Li X, Zhu J, et al. A Local Similarity-Preserving Framework for Nonlinear Dimensionality Reduction with Neural Networks[C]//International Conference on Database Systems for Advanced Applications. Springer, Cham, 2021: 376-391.
```


### Requirements
Before starting this project, you must install requirements below.

```
faiss==1.7.0
gensim==4.0.1
networkx==2.6.2
scikit-learn==0.24.2
```

Note：It's recommended that using conda to install faiss, and conda version need to update.

```
conda install faiss-cpu -c conda-forge
```

### Basic Usage

1. To run *vec2vec*  by terminal, execute the following command from the project home directory:
   ```
   python ./vec2vec/main.py --input ./vec2vec/data/train.bow
   ```
   
   You can check out the other options available by using:
   ```
   python ./vec2vec/main.py  --help
   ```

2. To run the vec2vec in your project, execute the following command:

   ```
   pip install vec2vec
   ```

### Input
Refer to the ./vec2vec/data/train.bow in the project.

### Output
The output are like below:

	************* The number of num_walks is : 5 *******************
	Matrix2vec p and q and topk: 1 1 10
	The shape of the input matrix: (2000, 13155)
	BuildNNGraphFromFAISS Finished in 0:00:03.305026 s.
	The shape of the adjmatrix is: (2000, 2000)
	Preprocess_transition_probs Finished in 0:00:00.902988 s.
	Random Walk Finished in 0:00:00.590795 s.
	Begin to train word2vec...
	Model Matrix2vec Finished in 0:00:09.073013 s.
	Accuracy：  [0.662 0.652 0.64  0.668]
	Accuracy: 0.6555 (+/- 0.0212)

### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to xiangwangcn@nudt.edu.cn.
