# Client-Level Differential Privacy via Adaptive Intermediary in Federated Medical Imaging
This is the PyTorch implemention of our MICCAI 2023 paper **Client-Level Differential Privacy via Adaptive Intermediary in Federated Medical Imaging**.
by [Meirui Jiang](https://github.com/MeiruiJiang), [Yuan Zhong](https://github.com/yzhong22), [Anjie Le](https://ale256.github.io/), [Xiaoxiao Li](https://xxlya.github.io/xiaoxiao/) and [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/).

## Abstract
> Despite recent progress in enhancing the privacy of federated learning (FL) via differential privacy (DP), the DP trade-off between privacy protection and performance is still underexplored for real-world medical use. In this paper, we propose to optimize the trade-off under the context of client-level DP, which focuses on privacy during communications. However, FL for medical imaging involves typically much fewer participants (hospitals) than other domains (e.g., mobile devices), thus ensuring clients be differentially private is much more challenging. To tackle this, we propose an adaptive intermediary strategy to improve performance without harming privacy. Specifically, we theoretically find splitting clients into sub-clients, which serve as intermediaries between hospitals and the server, can mitigate the noises introduced by DP without harming privacy. Our proposed approach is empirically evaluated on both classification and segmentation tasks using two public datasets, and its effectiveness is demonstrated with significant performance improvements and comprehensive analytical studies.


## Usage
### Setup
**Conda**

We recommend using conda to setup the environment, See the `requirements.yaml` for environment configuration 

If there is no conda installed on your PC, please find the installers from https://www.anaconda.com/products/individual

If you have already installed conda, please use the following commands.

```bash
conda env create -f environment.yaml
conda activate DP-FL
```
**Pip**
```bash
pip install -r requirements.txt
```



### Dataset 
#### ICH Classification
- Please download the dataset from [kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) and preprocess it follow this [notebook](https://www.kaggle.com/guiferviz/prepare-dataset-resizing-and-saving-as-png). You can also download the preprocessed the dataset from [here](https://drive.google.com/drive/folders/1bhe_0KvdxEli7-6ZrQ9ahaDPpSnvF4UW?usp=share_link).

#### Prostate MRI Segmentation
- Please refer to the [prostate MRI datasets](https://github.com/NVIDIA/NVFlare/tree/dev/examples/advanced/prostate) for details of data preparation. In the following, we assume the data has been saved to `../data_preparation/dataset_2D`


### Run
`fed_train.py` is the main file to run the federated experiments
Please refer to the following two command files for details, including all the experimental results in Table 1.
```bash
bash run_ich_exp.sh
bash run_prostate_exp.sh
```
## Citation
If this repository is useful for your research, please cite:

       @article{jiang2022clientDP,
         title={Client-Level Differential Privacy via Adaptive Intermediary in Federated Medical Imaging},
         author={Jiang, Meirui and Zhong, Yuan and Le, Anjie and Li, Xiaoxiao and Dou, Qi},
         journal={International Conference on Medical Image Computing and Computer Assisted Intervention},
         year={2023}
       }  

### Contact
For any questions, please contact 'mrjiang@cse.cuhk.edu.hk' or 'yuanzhongjr@gmail.com'