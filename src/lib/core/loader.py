import os
import copy
import random
import multiprocessing
from concurrent import futures

import mxnet as mx

from dataset.hm36 import *
from config import config

ex = futures.ThreadPoolExecutor(multiprocessing.cpu_count())

def get_batch_samples(dbs):
    data = []
    label = []
    folder = []

    DEBUG= False
    if DEBUG:
        for sample in dbs:
            targets = get_sample(sample)
            data.append(targets[0])
            label.append(targets[1])
            folder.append(targets[2])
    else:
        args = [{
                    'sample': sample
                } for sample in dbs]

        targets = ex.map(get_sample_worker, args)
        for target in targets:
            data.append(target[0])
            label.append(target[1])
            folder.append(target[2])

    return {
            'hm36data': data,   #32x5x32
            'hm36label': label, #32x5x48
            'hm36folder': folder
            }

def get_sample_worker(args):
    return get_sample(args['sample'])

def get_sample(sample):
    joints2d_list = copy.deepcopy(sample['pt_2d_list'])
    joints3d_list = copy.deepcopy(sample['pt_3d_list'])
    folder = sample['folder']

    assert len(joints2d_list) == len(joints3d_list)

    data  = []
    label = []
    for idx in range(len(joints2d_list)):
        data.append(joints2d_list[idx]) #5x32
        label.append(joints3d_list[idx]) #5x48

    return data, label, folder

def get_orgnized_DB(db, runmode, logger, action, meanstd):
    r'''orgnize the original dataset into 2d/3d sequences
    accoring to sequence length(Sliding Window Size) and
    sequence step(Sliding Window move step)'''

    seq_length = config.DATASET.seqLength
    cache_dir = './cache/orgnized/' if not config.DATASET.sameSample else './cache/sameSample/'
    if not os.path.exists(os.path.join(cache_dir)):
        os.makedirs(os.path.join(cache_dir))

    noise_prefix = 'noise_sigma%d_'%config.DATASET.sigma
    seq_step = 1 if runmode!='valid' else 64
    act_postfix = ('_act%s'%action) if action else ''
    cache_file = os.path.join(cache_dir, noise_prefix + 'HM36_%s_step%d_length%d'%(runmode, seq_step, seq_length) + act_postfix + '.pkl')

    if os.path.exists(cache_file):
        with open(cache_file,'rb') as fid:
            orgnized_db = pickle.load(fid)
        logger.info('LOADING.... {}, totally {} sequences'.format(cache_file, len(orgnized_db)))
        return orgnized_db

    logger.info('Orgnize data into sequence')
    DBsize = len(db)
    orgnized_db = []
    
    mean2d = meanstd['mean2d']
    std2d  = meanstd['std2d']
    mean3d = meanstd['mean3d']
    std3d  = meanstd['std3d']
    
    if config.DATASET.sameSample:
        for cur in np.arange(DBsize, step=seq_step):
            pt_2d_list = []
            pt_3d_list = []
            for _ in range(seq_length):
                pt_2d_list.append((db[cur]['joints_2d'].flatten()-mean2d)/std2d)
                pt_3d_list.append((db[cur]['joints_3d'].flatten()-mean3d)/std3d)
            orgnized_db.append({'pt_2d_list':pt_2d_list, 'pt_3d_list':pt_3d_list, 'folder':db[cur]['folder']})
    else:
        begin_idx = 0
        end_idx = begin_idx + seq_length
        while(end_idx <= DBsize):
            # if at the border of two folders
            if db[begin_idx]['folder'] != db[end_idx-1]['folder']:
                begin_idx = end_idx-1
                end_idx = begin_idx + seq_length
                continue

            pt_2d_list = []
            pt_3d_list = []
            for n_img in range(begin_idx, end_idx):
                pt_2d_list.append((db[n_img]['joints_2d'].flatten()-mean2d)/std2d)
                pt_3d_list.append((db[n_img]['joints_3d'].flatten()-mean3d)/std3d)

            pt_2d_list.reverse() #store the 2d data in reversed order
            orgnized_db.append({'pt_2d_list':pt_2d_list, 'pt_3d_list':pt_3d_list, 'folder':db[begin_idx]['folder']})
            begin_idx += seq_step
            end_idx = begin_idx + seq_length

    with open(cache_file, 'wb') as fid:
        pickle.dump(orgnized_db, fid, pickle.HIGHEST_PROTOCOL)
    logger.info('WRITING.... {}, totally {} sequences'.format(cache_file, len(orgnized_db)))
    return orgnized_db

class JointsDataIter(mx.io.DataIter):
    def __init__(self, db, runmode, data_names, label_names, shuffle, batch_size, logger, action=None):
        super(JointsDataIter, self).__init__()
        self.runmode = runmode
        # runmode:0-train,1-eval,2-test
        raw_db = None
        if runmode == 'train' or runmode == 'valid':
            raw_db = db.gt_db(logger)
        if runmode == 'test':
            raw_db = db.gt_db_actions(action, logger)

        # calculate mean&std from the entire training dataset,
        # thus the raw_db is only valid when runmmode==0&1
        # when runmode==test, only read mean&std from cache
        self.mean2d, self.std2d, self.mean3d, self.std3d = db.get_meanstd(raw_db, logger)
        self.db = get_orgnized_DB(raw_db, runmode, logger, action,
        {'mean2d':self.mean2d, 'std2d':self.std2d, 'mean3d':self.mean3d, 'std3d':self.std3d})

        self.size = len(self.db)
        self.index = np.arange(self.size)
        self.shuffle = shuffle
        self.joint_num = db.joint_num
        self.batch_size = batch_size

        self.cur = 0
        self.batch = None

        self.data_names = data_names
        self.label_names = label_names

        # status variable for synchronization between get_data and get_label
        self.data  = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()          # Reset and shuffle
        self.get_batch()

    def get_meanstd(self):
        return {'mean2d':self.mean2d, 'std2d':self.std2d, 'mean3d':self.mean3d, 'std3d':self.std3d}

    def get_size(self):
        return self.size

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_names, self.label)] if self.label_names else None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            data_batch = mx.io.DataBatch(data=self.data, label=self.label,
                                         pad=self.getpad(), index=self.getindex(),
                                         provide_data=self.provide_data, provide_label=self.provide_label)
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur                                                     # start index
        cur_to = min(cur_from + self.batch_size, self.size)                     # end index
        joints_db = [self.db[self.index[i]] for i in range(cur_from, cur_to)]   # fetch the data

        rst = get_batch_samples(joints_db)
        self.data  = [mx.nd.transpose(mx.nd.array(rst['hm36data']), axes=(1,0,2))]  #convert to sequence_size x batchsize x datasize
        self.label = [mx.nd.transpose(mx.nd.array(rst['hm36label']), axes=(1,0,2)), mx.nd.array(rst['hm36folder'])] #convert to sequence_size x batchsize x datasize
