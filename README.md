# Exploiting temporal information for 3D pose estimation

Reproduction of [Exploiting temporal information for 3D pose estimation](https://arxiv.org/abs/1711.08585v1</br>)<br/>
Original implement is [here](https://github.com/rayat137/Pose_3D)

TODO:
- [ ] Provide trained model
- [ ] Refine project

## Environment
python 3.7</br>
mxnet-cu90 1.4.0</br>
CUDA 9.0

## Dependency
``` 
pip install pyyaml
pip install scipy
pip install matplotlib
pip install easydict
``` 

## Dataset
1. [Baidu Disk](https://pan.baidu.com/s/1Qg4dH8PBXm8SzApI-uu0GA) (code: kfsm) or [Google Drive](https://drive.google.com/file/d/1wZynXUq91yECVRTFV8Tetvo271BXzxwI/view?usp=sharing) to download the HM3.6M annotation
2. Unzip data under *data* folder, and organize like this
```
${PROJECT_ROOT}
    `--data
        `--annot
            `--s_01_act_02_subact_01_ca_01
            `--s_01_act_02_subact_01_ca_02
            `-- ......
            `-- ......
            `-- ......
            `--s_11_act_16_subact_02_ca_04            
```

## How-to-use
```bash
usage: train.py/test.py [-h] --gpu GPU --root ROOT --dataset DATASET [--model MODEL]
                        [--debug DEBUG]

optional arguments:
  -h, --help         show this help message and exit
  --gpu GPU          GPUs to use, e.g. 0,1,2,3
  --root ROOT        /path/to/project/root/
  --dataset DATASET  /path/to/your/dataset/root/
  --model MODEL      /path/to/your/model/, to specify only when test
  --debug DEBUG      debug mode
```

**Train**: python train.py --root /project-root </br>

**Test**:  python test.py --root /project-root --model /model-path </br>

PS: You can modify default configurations in config.py.


## Results
Since I don't have 2D pose estimate results on HM3.6M, I just experiment with 2D ground truth as input.
My best result is **41.0mm** , slightly higher than 39.2mm reported by paper</br>

PS: LayerNorm is a component inside RNN cell. w/o=without

![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/doc/MPJPE_for_joints.png)

![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/doc/MPJPE_for_actions.png)

 ![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/doc/1.png)

 ![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/doc/2.png)

 ![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/doc/3.png)

 ![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/doc/4.png)
