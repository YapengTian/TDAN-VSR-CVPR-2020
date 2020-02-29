# TDAN-CVPR 2020 （The full code will be released soon!）

## Usage

Main dependencies: Python 3.6 and Pytorch-0.3.1 (https://pytorch.org/get-started/previous-versions/)

```bash
$ git clone https://github.com/YapengTian/TDAN-VSR
$ compile deformable convolution functions (may be optional): bash make.sh 
$ pip install -r requirements
$ python eval.py -t test_dataset_path
```

### Citation

If you find the code helpful in your resarch or work, please cite our paper:
```
@article{tian2018tdan,
  title={Tdan: Temporally deformable alignment network for video super-resolution},
  author={Tian, Yapeng and Zhang, Yulun and Fu, Yun and Xu, Chenliang},
  journal={arXiv preprint arXiv:1812.02898},
  year={2018}
}

@InProceedings{tian2020tdan,
  author={Yapeng Tian, Yulun Zhang, Yun Fu, and Chenliang Xu},
  title={TDAN: Temporally-Deformable Alignment Network for Video Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```
