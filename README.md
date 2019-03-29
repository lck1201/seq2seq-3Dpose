# Exploiting temporal information for 3D pose estimation

Reproduction of [Exploiting temporal information for 3D pose estimation](https://arxiv.org/abs/1711.08585v1</br>)<br/>
Original implement is [here](https://github.com/rayat137/Pose_3D)

TODO:
- [ ] Provide trained model
- [ ] Refine project

## Environment
python3.6</br>
mxnet-cu90 1.4.0</br>

## Dependency
``` 
pip install pyyaml
pip install scipy
pip install matplotlib
pip install easydict
``` 

## Dataset
[Here](https://pan.baidu.com/s/1Qg4dH8PBXm8SzApI-uu0GA) (code: kfsm) to download the HM3.6M annotation


## How-to-use
```bash
usage: train.py/test.py [-h] --gpu GPU --root ROOT --dataset DATASET [--model MODEL]
                        [--debug DEBUG]

optional arguments:
  -h, --help         show this help message and exit
  --gpu GPU          number of GPUs to use
  --root ROOT        /path/to/code/root/
  --dataset DATASET  /path/to/your/dataset/root/
  --model MODEL      /path/to/your/model/, to specify only when test
  --debug DEBUG      debug mode
```

**Train**: python train.py --data /datapath --root /project-path --gpu /gpu-to-use </br>

**Test**:  python test.py --data /data-path --root /project-path --gpu /gpu-to-use --model /model-path </br>

PS: You can modify default configurations in config.py.


## Results
Since I don't have 2D pose estimate results on HM3.6M, I just experiment with 2D ground truth as input.
My best result is **41.0mm** , higher than 39.2mm reported by paper</br>

PS: LayerNorm is a component inside RNN cell. w/o=without

![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/src/doc/MPJPE_for_joints.png)

![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/src/doc/MPJPE_for_actions.png)

 ![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/src/doc/1.png)

 ![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/src/doc/2.png)

 ![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/src/doc/3.png)

 ![image](https://github.com/lck1201/seq2seq-3Dpose/blob/master/src/doc/4.png)
