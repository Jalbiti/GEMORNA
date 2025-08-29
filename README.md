# GEMORNA: Deep generative models design mRNA sequences with enhanced translational capacity and stability


GEMORNA is an experimentally validated deep-learning package tailored for mRNA sequence design.
Leveraging generative AI models, GEMORNA is capable of designing superior mRNA sequences, including coding sequences (CDSs) and untranslated regions (UTRs), through zero-shot generation,
and supports both linear and circular mRNA design.
GEMORNA has been extensively verified in both cell-based and animal studies across a broad range of therapeutic applications. 
Results consistently show that GEMORNA-derived sequences outperform leading designs in expression, durability, and potency, establishing GEMORNA as a versatile platform to advance next-generation mRNA therapeutics and vaccines.


This repository hosts the most up-to-date version of GEMORNA, and we recommend using it as the primary source for software access and future updates.


# Dependencies
```
python==3.10
numpy==1.26.4
pandas==2.0.2
tqdm==4.66.5
torch==2.2.0
torchvision==0.17.0
torchaudio==2.2.0
torchtext==0.6.0
biopython==1.72
```

# Installation

We recommend running GEMORNA on Linux systems.
Please use the provided environment.yaml file to create a conda environment that matches the required dependencies before running the software.

```
sudo apt-get install git-lfs
git lfs install
git clone https://github.com/RainaBio/GEMORNA.git
cd GEMORNA/
conda env create -f environment.yaml
conda activate gemorna
```

# Run

## CDS generation
```
python src/generate.py --mode cds --ckpt_path checkpoints/gemorna_cds.pt --protein_seq ${protein_seq} 
```

## 5' UTR generation
```
python src/generate.py --mode 5utr --ckpt_path checkpoints/gemorna_5utr.pt --utr_length ${short, medium or long} 
```

## 3' UTR generation
```
python src/generate.py --mode 3utr --ckpt_path checkpoints/gemorna_3utr.pt --utr_length ${short, medium or long} 
```
Note:
options for utr_length are short, medium or long.


## 5'UTR prediction
```
python src/main_pred5UTR.py --ckpt_path checkpoints/5utr.pt --sequence ${5UTR_sequence}

```

## 3'UTR prediction
```
python src/main_pred3UTR.py --ckpt_path checkpoints/3utr.pt --sequence ${3UTR_sequence}
```

# Examples
```
python src/generate.py --mode cds --ckpt_path checkpoints/gemorna_cds.pt --protein_seq MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIF 
```

```
python src/generate.py --mode 5utr --ckpt_path checkpoints/gemorna_5utr.pt --utr_length short
```

```
python src/generate.py --mode 3utr --ckpt_path checkpoints/gemorna_3utr.pt --utr_length long
```

```
python src/main_pred5UTR.py --ckpt_path checkpoints/5utr.pt --sequence TACGTTTTGACCTTCGTTCATTTTG

```

```
python src/main_pred3UTR.py --ckpt_path checkpoints/3utr.pt --sequence TGTCCCCGGGTCTTCCAACGGACTGGCGTTGCCCCGGTTCACTGGGGACTGCCCTTGGGGTCTCGCTCACCTTCAGCACACATTATCGGGAGCAGTGTCTTCCATAATGT
```

## Citation

Please cite the following reference when referencing GEMORNA.

```
@article{zhang2025deep,
  title={Deep generative models design mRNA sequences with enhanced translational capacity and stability},
  author={Zhang, He and Liu, Hailong and Xu, Yushan and Huang, Haoran and Liu, Yiming and Wang, Jia and Qin, Yan and Wang, Haiyan and Ma, Lili and Xun, Zhiyuan and Hou, Xuzhuang and Lu, Timothy K and Cao, Jicong},
  journal={Science},
  year={2025},
}
```