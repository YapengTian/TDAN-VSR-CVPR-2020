# TDAN_VSR （The pre-trained model is not used model in the arxiv paper.）

## Usage

Main dependencies: Python 3.6 and Pytorch-0.3.1 (https://pytorch.org/get-started/previous-versions/)

```bash
$ git clone https://github.com/YapengTian/TDAN_VSR
$ compile deformable convolution functions (may be optional): bash make.sh 
$ pip install -r requirements.txt
$ run: python eval.py -t test_dataset_path -- it will save the 3000 frames into the "res" folder.
```
