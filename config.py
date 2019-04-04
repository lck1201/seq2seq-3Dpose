import yaml
import mxnet
import argparse
from easydict import EasyDict as edict

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='GPUs to use, e.g. 0,1,2,3',  default='0', type=str)
    parser.add_argument('--root', help='/path/to/code/root/',
                        default='/home/chuankang/code/seq2seq-3Dpose/', type=str)
    parser.add_argument('--dataset', help='/path/to/your/dataset/root/',
                        default='./data/', type=str)
    parser.add_argument('--model', help='/path/to/your/model/, to specify only when test', type=str)
    parser.add_argument('--debug', help='debug mode', default=False, type=str2bool)
    args, rest = parser.parse_known_args()
    return args

s_args = parse_args()

config = edict()

config.MXNET_VERSION = 'mxnet-cu90_' +  mxnet.__version__
config.block = 'Exploiting temporal information for 3D pose estimation'
config.saveModel_path = './output/model/'
config.train_log_path = './output/train-log/'
config.final_Model_path = ' '

config.DEBUG = False
config.useGPU = True
config.gpu = '0'

config.resume = False
config.CheckpointFile = ''
config.begin_epoch = 0
config.end_epoch = 100

# network-related config
config.NETWORK = edict()
config.NETWORK.hidden_dim = 1024
config.NETWORK.dropout1 = 0
config.NETWORK.dropout2 = 0.5
config.NETWORK.nJoints = 16

# train-related config
config.TRAIN = edict()
config.TRAIN.batchsize = 64
config.TRAIN.optimizer = 'adam'
config.TRAIN.lr = 1e-3
config.TRAIN.decay_rate = 0.96
config.TRAIN.SHUFFLE = True
config.TRAIN.UseMetric = False
config.TRAIN.w = 0 #cofficent for loss2

# dataset-related config
config.DATASET = edict()
config.DATASET.dbname = ['hm36']
config.DATASET.train_image_set = ['train']
config.DATASET.valid_image_set = ['valid']
config.DATASET.test_image_set  = ['test']
config.DATASET.root_path = []
config.DATASET.dataset_path = []
config.DATASET.seqLength = 2
config.DATASET.sameSample = False
config.DATASET.sigma = 0

# test-related config
config.TEST = edict()
config.TEST.batchsize = 64
config.TEST.isPA = False

def update_config(config_file):
    # exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_config_from_args(config, args):
    config.gpu = args.gpu
    config.DEBUG = args.debug
    config.DATASET.root_path = [args.root]
    config.DATASET.dataset_path = [args.dataset]

    return config

if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])

