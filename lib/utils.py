import os
from time import strftime
from time import localtime

import numpy as np
import mxnet as mx
from mxnet import ndarray as nd

from lib.dataset.hm36 import HM_act_idx, JntName

class LOG(object):
    def __init__(self, log_path, _isDebug):
        self.isDebug = _isDebug
        self.file = None
        if not self.isDebug:
            self.file = open(log_path,'w')

    def info(self, *args):
        ctn= ''
        for item in args:
            ctn += str(item) + ' '
        if not self.isDebug:
            print(strftime("[%Y-%m-%d %H:%M:%S] ", localtime()) + ctn, file=self.file)
            self.file.flush()
        print(strftime("[%Y-%m-%d %H:%M:%S] ", localtime()) + ctn)

    def kill(self):
        if self.file:
            self.file.close()

def create_logger(cfg, cfg_name=' '):
    # set up logger
    time_str = strftime('%Y-%m-%d_%H-%M')
    root_output_path = cfg.saveModel_path

    # model file save path
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

    # cfg_name = os.path.basename(cfg_name).split('.')[0]
    # model_path = os.path.join(root_output_path, '{}'.format(cfg_name))
    model_path = os.path.join(root_output_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    name = '{}_noise-sigma{}_seq{}_w{}_gpu{}'.format(time_str, cfg.DATASET.sigma, cfg.DATASET.seqLength, cfg.TRAIN.w, cfg.gpu)
    if cfg.DEBUG:
        final_model_path = os.path.join(model_path, 'Debug_' + name)
    else:
        final_model_path = os.path.join(model_path, name)
    if not os.path.exists(final_model_path):
        os.makedirs(final_model_path)

    # train logger
    # train_log_path = os.path.join(cfg.train_log_path, '{}'.format(cfg_name))
    train_log_path = os.path.join(cfg.train_log_path)
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)

    final_train_log_path = os.path.join(train_log_path, name + '.log')
    logger = LOG(final_train_log_path, cfg.DEBUG)
    print (final_train_log_path)

    return logger, final_model_path

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list


def saveModel(net, logger, config, isCKP=False, epoch=1):
    if isCKP:
        name = 'Checkpoint_epoch{}.params'.format(str(epoch))
    else:
        name = 'Model_epoch{}.params'.format(str(epoch))
    path = os.path.join(config.final_Model_path, name)
    net.collect_params().save(path)
    logger.info("Write Model/CheckPoint into", path)


def LogResult(logger, config, results):
    DBsize = [item[0] for item in results]
    Loss   = [item[1] for item in results]
    MPJPE  = [item[3] for item in results]
    XYZErr = [item[4] for item in results]
    JntErr = [item[5] for item in results]

    t_size, t_error = 0, 0
    t_xyz = np.zeros(3)
    t_jnt = np.zeros(16) if not config.TEST.isPA else np.zeros(17)
    logger.info("Procrustes Analysis:", config.TEST.isPA)
    for act, size, ls, t, mpjpe_err, xyz_err, jnt_e in zip(HM_act_idx, DBsize, Loss, MPJPE, XYZErr, JntErr):
        seqNum = size[0] / config.DATASET.seqLength  # one frame - one sequence
        t_size += seqNum
        t_error += seqNum * mpjpe_err
        t_xyz += seqNum * xyz_err
        t_jnt += seqNum * jnt_e
        logger.info("=========================================")
        logger.info("For action          : %02d" % act)
        logger.info("Sequence Size       : %d" % (seqNum))
        # logger.info("Single Forward Time : %.2f ms" % (1000 * t / seqNum))
        logger.info("L1/L2/TotalLoss     : %.2e / %.2e / %.2e" % (
        ls[0] / size[0], ls[1] / size[1], ls[0] / size[0] + config.TRAIN.w * ls[1] / size[1]))
        logger.info("MPJPE(17j)          : %.2f" % mpjpe_err)
        logger.info("X Y Z               : {}".format(" ".join(['%.1f' % x for x in xyz_err])))
        logger.info("-----Joint Error-----")
        for i in range(len(JntName)):
            logger.info("Joint %-10s Error: %.1f" % (JntName[i], jnt_e[i]))
        logger.info("=========================================\n")

    # Num of data
    DBs = np.array(DBsize)
    Tn1, Tn2 = DBs[:, 0].sum(), DBs[:, 1].sum()
    # L1 & L2
    Ls = np.array(Loss)
    L1s, L2s = Ls[:, 0].sum() / Tn1, Ls[:, 1].sum() / Tn2

    mean_error = t_error / t_size
    mean_xyz = t_xyz / t_size
    mean_jnterr = t_jnt / t_size
    logger.info("Total Sequences      : %d" % (t_size))
    logger.info("Mean XYZ             : {}".format(" ".join(['%.1f' % x for x in mean_xyz])))
    logger.info("L1/L2/TotalLoss      : %.2e / %.2e /%.2e" % (L1s, L2s, L1s + config.TRAIN.w * L2s))
    logger.info("MPJPE(17j)           : %.2f" % mean_error)
    for i in range(len(JntName)):
        logger.info("MEAN %-10s      : %.1f" % (JntName[i], mean_jnterr[i]))