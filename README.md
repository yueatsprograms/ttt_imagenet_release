Code release for [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://arxiv.org/abs/1909.13231).\
This code produces our results on ImageNet-C and ImageNet-Video-Robust. 
The CIFAR-10 results are produced by [this repository](https://github.com/yueatsprograms/ttt_cifar_release).

## Requirements
1. Our code requires pytorch version 1.0 or higher, with at least one modern GPU of adequate memory.
2. We ran our code with python 3.7. Compatibility with python 2 is possible maybe with some modifications.
3. Most of the packages used should be included with [anaconda](https://www.anaconda.com/distribution/), 
except maybe two small utilities:
	- [tqdm](https://github.com/tqdm/tqdm), which we installed with `conda install tqdm`.
	- [colorama](https://pypi.org/project/colorama/), which we installed with `conda install colorama`.
4. Download the two datasets into the same folder:
	- [ImageNet-C](https://arxiv.org/abs/1903.12261) (Hendrycks and Dietterich) 
from [this repository](https://github.com/hendrycks/robustness).
	- [ImageNet-Video-Robust](https://arxiv.org/abs/1906.02168) (Shankar et al.) 
from [this repository](https://github.com/modestyachts/imagenet-vid-robust-example).


## Steps
1. Clone our repository with
`git clone https://github.com/yueatsprograms/ttt_imagenet_release`.
2. Inside the repository, set the data folder to where the datasets are stored by editing:
	- `--dataroot` argument in `main.py`.
	- `--dataroot` argument in `test_video.py`.
	- `dataroot` variable in `script_test.py`.
3. Run `script.sh`.
4. The results are stored in the respective folders in `results/`.
5. Once everything is finished, the results can be compiled and visualized with the following utilities:
	- `show_table.py` parses the results into tables and prints them.
	- `show_plot.py` makes bar plots like those in our paper, and prints the tables in latex format; requires first running `show_table.py`.
	
