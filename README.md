# PyHessian

## Introduction
PyHessian is a pytorch library for Hessian based analysis of neural network models that could be used in conjunction with PyTorch. The library currently supports utility functions to compute (i) top n largest eigenvlaues, (ii) the trace fo the entire Hessian matrix, (iii) the estimated emperical eigenvalues density, of different neural network models.

## Usage
The usage of PyHessian is very simple. Clone the PyHessian library to your local file is all your need:
```
git clone https://github.com/amirgholami/PyHessian.git
```

Before running the Hessian code, we need a pre-trained model (it can be just in initilization phase). Here, we provide a training file to train ResNet20 model on Cifar-10 dataset:
```
export CUDA_VISIBLE_DEVICES=0; python training.py [--batch-size] [--test-batch-size] [--epochs] [--lr] [--lr-decay] [--lr-decay-epoch] [--seed] [--weight-decay] [--batch-norm] [--residual] [--cuda] [--saving-folder]

optional arguments:
--batch-size                training batch size (default: 128)
--test-batch-size           testing batch size (default:256)
--epochs                    total number of training epochs (default: 180)
--lr                        initial learning rate (default: 0.1)
--lr-decay                  learning rate decay ratio (default: 0.1)
--lr-decay-epoch            epoch for the learning rate decaying (default: 80, 120)
--seed                      used to reproduce the results (default: 1)
--weight-decay              weight decay value (default: 5e-4)
--batch-norm                do we need batch norm in ResNet or not (default: True)
--residual                  do we need residual connection or not (default: True)
--cuda                      do we use gpu or not (default: True)
--saving-folder             saving path of the final checkpoint (default: checkpoints/)
```

After get the checkpoint (actually, we include one in the current repo), we can run the following code to get top-1 eigenvalue, the trace, as well as the density estimation:
```
export CUDA_VISIBLE_DEVICES=0; python example_pyhessian_analysis.py [--batch-size] [--hessian-batch-size] [--seed] [--batch-norm] [--residual] [--cuda] [--resume]

optional arguments:
--mini-hessian-batch-size   mini hessian batch size (default: 200)
--hessian-batch-size        hessian batch size (default:200)
--seed                      used to reproduce the results (default: 1)
--batch-norm                do we need batch norm in ResNet or not (default: True)
--residual                  do we need residual connection or not (default: True)
--cuda                      do we use gpu or not (default: True)
--resume                    resume path of the checkpoint (default: none, much be filled by user)
```

Note that: there are lots of parameters inside example_pyhessian_analysis.py that we did not include in parser, e.g. top_n (used to get the top n eigenvalues of the neural network). Please look at the hessian.py inside pyhessian folder for reference. 

After we compute the statistics, we will get all the results in the result.txt file. 

## Citation
PyHessian has been developed as part of the following papers. We appreciate it if you would please cite the following if you found the library useful for your work:

* Z. Yao, A. Gholami, K Keutzer, M. Mahoney. PyHessian:  Neural Networks Through the Lens of the Hessian, under review
* Z. Yao, A. Gholami, Q. Lei, K. Keutzer, M. W. Mahoney. Hessian-based Analysis of Large Batch Training and Robustness to Adversaries, NIPS'18 [PDF](https://arxiv.org/pdf/1802.08241)

