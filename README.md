# TCA: Test-time Correlation Alignment
__This repo is officical PyTorch implement of ['Test-time Correlation Alignment' (ICML 2025)](https://arxiv.org/abs/2505.00533)__  
This codebase is mainly based on [TSD](https://github.com/SakurajimaMaiii/TSD) and [AETTA](https://github.com/taeckyung/AETTA).  
## Dependence
We use `python==3.8.13`, other packages including:
```
torch==1.12.0+cu113
torchvision==0.13.0+cu113
numpy==1.24.4
pandas==2.0.3
tqdm==4.66.2
timm==0.9.16
scikit-learn==1.3.2 
pillow==10.3.0
```
We also share our python environment that contains all required python packages. Please refer to the `./TCA.yml` file.  
You can import our environment using conda:
```
conda env create -f TCA.yml -n TCA
```
## Dataset
Download __PACS__ and __OfficeHome__ datasets used in our paper from:  
[PACS](https://huggingface.co/datasets/flwrlabs/pacs)  
[OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html) 
Download them from the above links, and organize them as follows.  
```
|-your_data_dir
  |-PACS
    |-art_painting
    |-cartoon
    |-photo
    |-sketch
  |-OfficeHome
    |-Art
    |-Clipart
    |-Product
    |-RealWorld
```
To download the __CIFAR10/CIFAR10-C__ and __CIFAR100/CIFAR100-C__ datasets ,run the following commands:
```
$. download_cifar10c.sh        #download CIFAR10/CIFAR10-C datasets
$. download_cifar100c.sh       #download CIFAR100/CIFAR100-C datasets
```

## Train source model
Please use `train.py` to train the source model. For example:
```
python train.py --dataset PACS \
                --data_dir your_data_dir \
                --opt_type Adam \
                --lr 5e-5 \
                --max_epoch 50 \
                --net resnet18 \
                --test_envs 0  \
```
Change `--dataset PACS` for other datasets, such as `office-home`,`VLCS`,`DomainNet`, `CIFAR-10`, `CIFAR-100`.  
Set `--net` to use different backbones, such as `resnet50`, `ViT-B16`.  
Set `--test_envs 0` to change the target domain.  
For CIFAR-10 and CIFAR-100, there is no need to set the `--data_dir` and `--test_envs` .
## Test time adaptation
For domain datasets such as _PACS_ and _OfficeHome_, run the following code:
```
python unsupervise_adapt.py --dataset PACS \
                            --data_dir your_data_dir \
                            --pretrain_dir your_pretrain_model_dir \
                            --adapt_alg SOURCE \
                            --net resnet18 \
                            --test_envs 0  \
                            --lr 1e-3 \
                            --Add_TCA True \
                            --filter_K_TCA 20           
```

For corrupted datasets such as _CIFAR10-C_ and _CIFAR100-C_, run the following code:  
```
python unsupervise_adapt_corrupted.py --dataset CIFAR-10 \
                                      --data_dir your_data_dir \
                                      --pretrain_dir your_pretrain_model_dir \
                                      --adapt_alg SOURCE \ 
                                      --net resnet18 \
                                      --lr 1e-4 \
                                      --Add_TCA True \
                                      --filter_K_TCA 20 
```
Change `--adapt_alg SOURCE` to use different methods of test time adaptation, e.g.  `BN`, `Tent`, `TSD`.  
`--pretrain_dir` denotes the path of source model, e.g. `./train_outputs/model.pkl`.  

For the TCA method, you can modify the `--Add_TCA` parameter to `True` or `False` to determine whether TCA is included in the TTA process.  Our `LinearTCA` method defaults to the combination of SOURCE and TCA. TCA can also be combined with any TTA method to form a `LinearTCA+` method.  

For parameter selection in the TCA process, you can focus on modifying the `--filter_K_TCA`. For optimal selection of _filter_K_TCA_, please refer to our paper. If you wish to further fine-tune the TCA process, you can also consider adjusting the `--W_num_iterations` and `--W_lr` parameters.

**Note**: During the Test Time Adaptation phase, integrating any given TTA method with TCA requires the following adjustment to the TTA method.
```
class TTA_Method(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.featurizer = model.featurizer
        self.classifier = model.classifier

    def forward(self, x):
        with torch.no_grad():
          z = self.featurizer(x) #Supply an additional output of the model’s embeddings.
        outputs = model.predict(x)
        return z, p  
```
## Tested Environment
We tested our code in the environment described below.
```
OS: Ubuntu 18.04.6 LTS
GPU: NVIDIA GeForce RTX 4090
GPU Driver Version: 535.129.03
CUDA Version: 12.2
```
