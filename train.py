import pprint
import time
import os

import mxnet as mx
from mxnet import nd
from mxnet import init
from mxnet import gluon

from lib.core.metric import MPJPEMetric
from lib.core.loss import MeanSquareLoss
from lib.core.loader import JointsDataIter
from lib.core.network import get_net
from lib.core.get_optimizer import get_optimizer
from lib.dataset.hm36 import hm36, torso_head, limb_mid, limb_terminal
from lib.utils import *
from lib.net_module import *

from config import config, gen_config, s_args, update_config_from_args
config = update_config_from_args(config, s_args)

def main():
    # Parse config and mkdir output
    logger, final_Model_path = create_logger(config)
    config.final_Model_path = final_Model_path
    gen_config(os.path.join(final_Model_path, 'hyperParams.yaml'))
    logger.info('Training config:{}\n'.format(pprint.pformat(config)))

    # define context
    if config.useGPU:
        ctx = [mx.gpu(int(i)) for i in config.gpu.split(',')]
    logger.info("Using context:", ctx)

    # dataset, generate trainset/ validation set
    train_imdbs = []
    valid_imdbs = []
    for i in range(len(config.DATASET.train_image_set)):
        logger.info("Construct Dataset:", config.DATASET.dbname[i], ", Dataset Path:", config.DATASET.dataset_path[i])
        train_imdbs.append(eval(config.DATASET.dbname[i])(config.DATASET.train_image_set[i],
                                                          config.DATASET.root_path[i],
                                                          config.DATASET.dataset_path[i]))
        valid_imdbs.append(eval(config.DATASET.dbname[i])(config.DATASET.valid_image_set[i],
                                                          config.DATASET.root_path[i],
                                                          config.DATASET.dataset_path[i]))
    data_names  = ['hm36data']
    label_names = ['hm36label']
    train_data_iter = JointsDataIter(train_imdbs[0], runmode='train',
                                    data_names = data_names, label_names=label_names,
                                    shuffle=True, batch_size = len(ctx)*config.TRAIN.batchsize, logger=logger)
    valid_data_iter = JointsDataIter(valid_imdbs[0], runmode='valid',
                                    data_names = data_names, label_names=label_names,
                                    shuffle=False, batch_size = len(ctx)*config.TRAIN.batchsize, logger=logger)

    assert train_data_iter.get_meanstd()['mean3d'].all() == valid_data_iter.get_meanstd()['mean3d'].all()

    # network
    net = get_net(config)
    if config.resume:
        ckp_file = config.CheckpointFile
        net.collect_params().load(ckp_file, ctx=ctx)
        logger.info("Resume from:",ckp_file)
    else:
        net.initialize(init=init.Xavier(), ctx=ctx)

    logger.info(net)

    # define loss and metric
    mean3d = train_data_iter.get_meanstd()['mean3d']
    std3d  = train_data_iter.get_meanstd()['std3d']

    loss         = MeanSquareLoss()
    train_metric = MPJPEMetric('train_metric', mean3d, std3d)
    eval_metric  = MPJPEMetric('valid_metric', mean3d, std3d)

    # optimizer
    optimizer, optimizer_params = get_optimizer(config, ctx)

    # train and valid
    TrainDBsize = train_data_iter.get_size()
    ValidDBsize = valid_data_iter.get_size()
    logger.info("Train Sequence size:", TrainDBsize, "Valid Sequence size:",ValidDBsize)

    if not isinstance(train_data_iter, mx.io.PrefetchingIter):
        train_data_iter = mx.io.PrefetchingIter(train_data_iter)

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    for epoch in range(config.begin_epoch, config.end_epoch):
        trainNet(net, trainer, train_data_iter, loss, train_metric, epoch, config, logger=logger, ctx=ctx)
        validNet(net, valid_data_iter, loss, eval_metric, epoch, config, logger=logger, ctx=ctx)
        logger.info(" ")
    logger.kill()

if __name__ == '__main__':
    main()