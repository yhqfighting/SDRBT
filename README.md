# SDRBT:Patient deep spatio-temporal encoding and Medication substructure mapping for safe medication recommendation

## 1. Folder Specification


- `data/`
  - `drug-atc.csv`, `ndc2atc_level4.csv`, `ndc2rxnorm_mapping.txt`: mapping files for drug code transformation
  - `idx2ndc.pkl`: It maps ATC-4 code to rxnorm code and then query to drugbank.
  - `idx2drug.pkl`: Drug ID (we use ATC-4 level code to represent drug ID) to drug SMILES string dictionary
  - `voc_final.pkl`: diag/prod/med index to code dictionary
  - `ddi_A_final.pkl`: ddi adjacency matrix
  - `ddi_matrix_H.pkl`: H mask structure (This file is created by ddi_mask_H.py)
  - `records_final.pkl`: The final diagnosis-procedure-medication EHR records of each patient. Due to policy reasons, we are unable to provide processed data. Users are asked to process it themselves according to the instructions in the next section
  - `ddi_mask_H.py`: The python script responsible for generating `ddi_mask_H.pkl` and `substructure_smiles.pkl`.
  - `processing.py`: The python script responsible for generating `voc_final.pkl`, `records_final.pkl`, and `ddi_A_final.pkl`   

- `src/` folder contains all the source code.
  - `modules/`: Code for model definition.
  - `util.py`: Code for metric calculations and some data preparation.
  - `training.py`: Code for the functions used in training and evaluation.
  - `main.py`: Train or evaluate our Model.
 
- `saved/` 
  - `trained_model`:  test example, a model we have trained. Users can directly check using the test mode
  - `parameter_report.txt`: Log file containing all parameters
  
**Note1:** `data/` only contains part of the data. See the [Data processing] section for more details.

**Note2:** Due to some relatively complex environment dependencies during the causal graph generation phase, for the convenience of users in studying or validating our work, we have submitted a file named `causal_construction_easyuse.py`. This file can be used in conjunction with the already generated causal graphs, replacing the `causal_construction.py` file. While this method is more convenient, we strongly recommend researchers to retrain the causal graphs to ensure rigor.

## 2. Operation

## 2.1 Experimental Environment 

The experimental environment used in this study is as follows:

```bash 
Operating system: Ubuntu 20.04
CPU usage: 22 vCPU AMD EPYC 7T83 64-Core Processor.
GPU requirements: 24GB NVIDIA RTX4090 GPU.
Environment dependencies: PyTorch 2.0.0 and CUDA 11.8.
```
        
## 2.2 Package Dependency

Please install the environment according to the following version

```bash
absl-py                             2.1.0
astunparse                          1.6.3
beam                                0.6.0
beartype                            0.18.5
block-recurrent-transformer-pytorch 0.4.3
cachetools                          5.3.3
certifi                             2024.2.2
charset-normalizer                  3.3.2
click                               8.1.7
cmake                               3.29.2
colorama                            0.4.6
cycler                              0.12.1
dill                                0.3.8
dnc                                 1.1.0
einops                              0.8.0
et-xmlfile                          1.1.0
faiss-gpu                           1.7.2
filelock                            3.14.0
flann                               1.6.13
flatbuffers                         1.12
fonttools                           4.51.0
fsspec                              2024.3.1
gast                                0.4.0
google-auth                         2.29.0
google-auth-oauthlib                0.4.6
google-pasta                        0.2.0
graph-transformer-pytorch           0.1.1
greenlet                            3.0.3
grpcio                              1.63.0
h5py                                3.11.0
huggingface-hub                     0.17.3
idna                                3.7
imageio                             2.34.1
Jinja2                              3.1.4
joblib                              1.4.2
keras                               2.9.0
Keras-Applications                  1.0.8
Keras-Preprocessing                 1.1.2
kiwisolver                          1.4.5
lazy_loader                         0.4
libclang                            18.1.1
lit                                 18.1.4
littleutils                         0.2.2
loss                                0.1.2
Markdown                            3.6
markdown-it-py                      3.0.0
MarkupSafe                          2.1.5
matplotlib                          3.5.3
mdurl                               0.1.2
memorizing-transformers-pytorch     0.4.1
ml-dtypes                           0.4.0
modules                             1.0.0
mpmath                              1.3.0
munkres                             1.1.4
namex                               0.0.8
networkx                            3.3
nltk                                3.8.1
numpy                               1.25.2
nvidia-cublas-cu11                  11.10.3.66
nvidia-cuda-cupti-cu11              11.7.101
nvidia-cuda-nvrtc-cu11              11.7.99
nvidia-cuda-runtime-cu11            11.7.99
nvidia-cudnn-cu11                   8.5.0.96
nvidia-cufft-cu11                   10.9.0.58
nvidia-curand-cu11                  10.2.10.91
nvidia-cusolver-cu11                11.4.0.1
nvidia-cusparse-cu11                11.7.4.91
nvidia-nccl-cu11                    2.14.3
nvidia-nvtx-cu11                    11.7.91
oauthlib                            3.2.2
ogb                                 1.3.6
opencv-contrib-python               4.9.0.80
openpyxl                            3.1.2
opt-einsum                          3.3.0
optree                              0.11.0
outdated                            0.2.2
packaging                           24.0
pandas                              2.0.3
pillow                              10.3.0
pip                                 24.0
protobuf                            3.19.6
psutil                              5.9.8
pyasn1                              0.6.0
pyasn1_modules                      0.4.0
pyparsing                           3.1.2
python-dateutil                     2.9.0.post0
pytz                                2024.1
PyYAML                              6.0.1
rdkit                               2023.9.6
regex                               2024.4.28
requests                            2.31.0
requests-oauthlib                   2.0.0
rotary-embedding-torch              0.6.0
rsa                                 4.9
safetensors                         0.4.3
scikit-image                        0.23.2
scikit-learn                        1.3.0
scipy                               1.11.1
setuptools                          69.5.1
six                                 1.16.0
SQLAlchemy                          2.0.30
sympy                               1.12
tensorboard                         2.9.1
tensorboard-data-server             0.6.1
tensorboard-plugin-wit              1.8.1
tensorflow                          2.9.0
tensorflow-estimator                2.9.0
tensorflow-io-gcs-filesystem        0.37.0
termcolor                           2.4.0
threadpoolctl                       3.5.0
tifffile                            2024.5.3
tokenizers                          0.14.1
torch                               2.0.1
torch_geometric                     2.4.0
torch_scatter                       2.1.2
torchaudio                          2.0.2
torchtyping                         0.1.4
torchvision                         0.15.2
tqdm                                4.66.4
transformers                        4.34.1
triton                              2.0.0
typeguard                           4.2.1
typing_extensions                   4.11.0
tzdata                              2024.1
unicodedata2                        15.1.0
urllib3                             2.2.1
Werkzeug                            3.0.3
wheel                               0.43.0
wrapt                               1.16.0
x-transformers                      1.30.0
```

## 2.2 Data Processing

1.Since GitHub has a 100MB file size limit, we can only upload a portion of the data. To make reproduction more convenient, we have provided a compressed file containing all the data for download on Google Drive.

https://drive.google.com/file/d/1B_IniWXZvwsN2tjEcLuRm927C8peDFJD/view?usp=sharing

2.Readers can directly use the data we provide or regenerate the data using the code below.

```bash
python data/processing.py
python data/ddi_mask_H.py
python src/Relevance_construction.py
```

### 2.3 Run the Code

```bash
python src/main.py
```

## 3. Citation & Acknowledgement
We are grateful to everyone who contributed to this project.

If the code and the paper are useful for you, it is appreciable to cite our paper.
