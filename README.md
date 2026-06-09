# DPCDSR:Diffusion Prior-guided Cross-Domain Sequential Recommendation

## Environments

- Python 3.11
- Pytorch

## Dataset
Due to size limitations, the original files (available for download at [Amazon review data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) and [HVIDEO dataset](https://bitbucket.org/Catherine_Ma/pinet_sigir2019/src/master/HVIDEO/)) are not provided. Please download them at [Baidu Netdisk](https://pan.baidu.com/s/17k9qpu8iiK-gik0sltMo7Q?pwd=4fxe)The dataset is located in the `dataset/` folder and contains the three real-world datasets.



## Example to run the code
Train and evaluate the model (you are strongly recommended to run the program on a machine with a GPU):

python train_diffusion_hgn.py
python main.py --use_diffusion
