# HTPS
Source code of SIGIR2020 Long Paper "Encoding History with Context-aware Representation Learning for Personalized Search". Due to two sub-models QDM and PLM have similar structures, we release the code of QDM model here which has a more classic hierarchical transformer structure.

## Requirements
I test the code with the following packages. Other versions may also work, but I'm not sure. <br>
- Python 3.6 <br>
- Pytorch 1.3.1 (with GPU support)

Data processing and train the model.
```
python dataset_new.py
```

## Citations
If you use the code, please cite the following paper:  
```
@inproceedings{ZhouDW20,
  author    = {Yujia Zhou and
               Zhicheng Dou and
               Ji{-}Rong Wen},
  title     = {Encoding History with Context-aware Representation Learning for Personalized
               Search},
  booktitle = {{SIGIR}},
  pages     = {1111--1120},
  publisher = {{ACM}},
  year      = {2020}
}
