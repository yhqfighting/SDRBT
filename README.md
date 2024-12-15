# SDRBT:Patient deep spatio-temporal encoding and Medication substructure mapping for safe medication recommendation

## 1. Folder Specification


- `data/`
  - `input/` 
    - `drug-atc.csv`, `ndc2atc_level4.csv`, `ndc2rxnorm_mapping.txt`: mapping files for drug code transformation
    - `idx2ndc.pkl`: It maps ATC-4 code to rxnorm code and then query to drugbank.
    - `idx2drug.pkl`: Drug ID (we use ATC-4 level code to represent drug ID) to drug SMILES string dictionary
      
  - `output/`
    - `voc_final.pkl`: diag/prod/med index to code dictionary
    - `ddi_A_final.pkl`: ddi adjacency matrix
    - `ddi_matrix_H.pkl`: H mask structure (This file is created by ddi_mask_H.py)
    - `records_final.pkl`: The final diagnosis-procedure-medication EHR records of each patient. Due to policy reasons, we are unable to provide processed data. Users are asked to process it themselves according to the instructions in the next section
      
  - `graphs/`
    - `causal_graph.pkl`: casual graphs in DAG form
    - `Diag_Med_causal_effect.pkl`,`Proc_Med_casual_effect.pkl`: causal effects between diag/proc and med
    
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
absl-py                      2.1.0
astunparse                   1.6.3
beartype                     0.18.5
cachetools                   5.3.3
causal-learn                 0.1.3.8
certifi                      2024.2.2
charset-normalizer           2.1.1
clarabel                     0.9.0
click                        8.1.7
colorama                     0.4.6
cvxpy                        1.5.3
cycler                       0.11.0
Cython                       3.0.11
dill                         0.3.4
dnc                          1.1.0
dowhy                        0.11.1
ecos                         2.0.14
einops                       0.8.0
et-xmlfile                   1.1.0
filelock                     3.9.0
flann                        1.6.13
flatbuffers                  1.12
fonttools                    4.42.0
fsspec                       2023.9.2
gast                         0.4.0
google-auth                  2.29.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
graph-transformer-pytorch    0.1.1
graphviz                     0.20.3
greenlet                     2.0.2
grpcio                       1.62.1
h5py                         3.11.0
huggingface-hub              0.17.3
idna                         3.4
imageio                      2.34.0
Jinja2                       3.1.2
joblib                       1.3.2
keras                        2.9.0
Keras-Applications           1.0.8
Keras-Preprocessing          1.1.2
kiwisolver                   1.4.4
lazy_loader                  0.4
libclang                     18.1.1
littleutils                  0.2.2
loss                         0.1.2
Markdown                     3.6
markdown-it-py               3.0.0
MarkupSafe                   2.1.2
matplotlib                   3.5.3
mdurl                        0.1.2
ml-dtypes                    0.3.2
modules                      1.0.0
mpmath                       1.2.1
munkres                      1.1.4
namex                        0.0.7
networkx                     3.0
nltk                         3.8.1
numpy                        1.25.2
oauthlib                     3.2.2
ogb                          1.3.6
opencv-contrib-python        4.9.0.80
openpyxl                     3.1.2
opt-einsum                   3.3.0
optree                       0.11.0
osqp                         0.6.7.post3
outdated                     0.2.2
packaging                    23.1
pandas                       2.0.3
patsy                        0.5.6
Pillow                       9.3.0
pip                          23.2.1
protobuf                     3.19.6
psutil                       5.9.6
pyasn1                       0.6.0
pyasn1_modules               0.4.0
pycairo                      1.24.0
pydot                        3.0.2
Pygments                     2.17.2
pyparsing                    3.1.1
python-dateutil              2.8.2
pytz                         2023.3
PyYAML                       6.0.1
qdldl                        0.1.7.post4
rdkit-pypi                   2022.9.5
regex                        2023.10.3
reportlab                    3.6.12
requests                     2.28.1
requests-oauthlib            2.0.0
rich                         13.7.1
rotary-embedding-torch       0.5.3
rsa                          4.9
safetensors                  0.4.0
scikit-image                 0.23.1
scikit-learn                 1.3.0
scipy                        1.11.1
scs                          3.2.7
seaborn                      0.13.2
setuptools                   68.0.0
six                          1.16.0
SQLAlchemy                   2.0.19
statsmodels                  0.14.4
sympy                        1.11.1
tensorboard                  2.9.1
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow                   2.9.0
tensorflow-estimator         2.9.0
tensorflow-intel             2.16.1
tensorflow-io-gcs-filesystem 0.31.0
termcolor                    2.4.0
threadpoolctl                3.2.0
tifffile                     2024.2.12
tokenizers                   0.14.1
torch                        2.0.1+cu117
torch_geometric              2.4.0
torch-scatter                2.1.2
torchaudio                   2.0.2+cu117
torchtyping                  0.1.4
torchvision                  0.15.2+cu117
tqdm                         4.66.1
transformers                 4.34.1
typeguard                    4.2.1
typing_extensions            4.11.0
tzdata                       2023.3
unicodedata2                 15.0.0
urllib3                      1.26.13
Werkzeug                     3.0.2
wheel                        0.41.1
wrapt                        1.16.0
x-transformers               1.30.0
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
