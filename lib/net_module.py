import time

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd

from lib.utils import saveModel

def trainNet(net, trainer, train_data, loss, train_metric, epoch, config, logger, ctx):
    if not logger:
        assert False, 'require a logger'

    train_data.reset()  # reset and re-shuffle
    if train_metric:
        train_metric.reset()

    trainer.set_learning_rate(config.TRAIN.lr * pow(config.TRAIN.decay_rate, epoch))

    w = config.TRAIN.w
    batchsize = config.TRAIN.batchsize
    UseMetric = config.TRAIN.UseMetric
    seqLength = config.DATASET.seqLength
    nJoints = config.NETWORK.nJoints

    loss1, loss2, n1, n2 = [0] * len(ctx), [0] * len(ctx), 0.000001, 0.000001
    RecordTime = {'load': 0, 'forward': 0, 'backward': 0, 'post': 0}

    for batch_i, batch in enumerate(train_data):
        beginT = time.time()
        data_list = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=1)
        label_list = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx,
                                                batch_axis=1)  # [[seqLength x 64 x 48] , 4]
        RecordTime['load'] += time.time() - beginT

        # forward
        beginT = time.time()
        Ls, Ls1, Ls2, output_list = [], [], [], []
        with autograd.record():
            for data, label, cx in zip(data_list, label_list, ctx):
                initial_state = [nd.zeros(shape=(batchsize, config.NETWORK.hidden_dim), ctx=cx) for _ in range(2)]
                start_token = nd.ones(shape=(batchsize, 3 * nJoints), ctx=cx)
                preds = net(data, initial_state, start_token)
                output_list.append(preds)  # pred=[5, 64x48]

                L1, L2 = 0, 0
                for pd, lb in zip(preds, label):
                    L1 = L1 + loss(pd, lb)
                if seqLength > 1:
                    for i in range(1, seqLength):
                        deltaP = preds[i] - preds[i - 1]
                        deltaG = label[i] - label[i - 1]
                        L2 = L2 + loss(deltaP, deltaG)
                Ls1.append(L1)
                Ls2.append(L2) if seqLength > 1 else Ls2.append(nd.zeros(1))
                Ls.append(L1 + w * L2)
        RecordTime['forward'] += time.time() - beginT

        # backward
        beginT = time.time()
        for L in Ls:
            L.backward()
        trainer.step(len(ctx) * batchsize)
        RecordTime['backward'] += time.time() - beginT

        beginT = time.time()
        # number
        n1 = n1 + len(ctx) * batchsize * seqLength
        n2 = n2 + len(ctx) * batchsize * (seqLength - 1)

        # loss
        for i in range(len(loss1)):
            loss1[i] += Ls1[i]
            loss2[i] += Ls2[i]

        # metric, save time
        if UseMetric:
            for pred_batch, label_batch in zip(output_list, label_list):  # for each timestamp
                for t_pred, t_label in zip(pred_batch, label_batch):
                    train_metric.update(t_label, t_pred)
        RecordTime['post'] += time.time() - beginT

    totalT = nd.array([RecordTime[k] for k in RecordTime]).sum().asscalar()
    for key in RecordTime:
        print("%-s: %.1fs %.1f%% " % (key, RecordTime[key], RecordTime[key] / totalT * 100), end=" ")
    print(" ")

    nd.waitall()
    loss1 = sum([item.sum().asscalar() for item in loss1])
    loss2 = sum([item.sum().asscalar() for item in loss2])
    TotalLoss = loss1 / n1 + w * loss2 / n2
    MPJPE = train_metric.get()[-1].sum(axis=0).asscalar() / 17 if UseMetric else 0

    logger.info("TRAIN - Epoch:%2d LR:%.2e Loss1:%.2e Loss2(%2d):%.2e TotalLoss:%.2e MPJPE:%.1f" %
                (epoch + 1, trainer.learning_rate, loss1 / n1, w, loss2 / n2, TotalLoss, MPJPE))

    if ((epoch + 1) % (config.end_epoch / 4) == 0 or epoch == 0):  # save checkpoint
        saveModel(net, logger, config, isCKP=True, epoch=epoch + 1)
    if (epoch + 1 == config.end_epoch):  # save final model
        saveModel(net, logger, config, isCKP=False, epoch=epoch + 1)


def validNet(net, valid_data, loss, eval_metric, epoch, config, logger, ctx):
    if not logger:
        assert False, 'require a logger'

    valid_data.reset()
    if eval_metric:
        eval_metric.reset()

    w = config.TRAIN.w
    batchsize = config.TRAIN.batchsize
    UseMetric = config.TRAIN.UseMetric
    seqLength = config.DATASET.seqLength
    nJoints = config.NETWORK.nJoints

    loss1, loss2, n1, n2 = [0] * len(ctx), [0] * len(ctx), 0.000001, 0.000001
    for batch_i, batch in enumerate(valid_data):
        data_list = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=1)
        label_list = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=1)

        Ls1, Ls2, output_list = [], [], []
        # forward
        for data, label, cx in zip(data_list, label_list, ctx):
            initial_state = [nd.zeros(shape=(batchsize, config.NETWORK.hidden_dim), ctx=cx) for _ in range(2)]
            start_token = nd.ones(shape=(batchsize, 3 * nJoints), ctx=cx)
            preds = net(data, initial_state, start_token)
            output_list.append(preds)  # pred=[seqLength, 64x48]

            L1, L2 = 0, 0
            for pd, lb in zip(preds, label):
                L1 = L1 + loss(pd, lb)
            if seqLength > 1:
                for i in range(1, seqLength):
                    deltaP = preds[i] - preds[i - 1]
                    deltaG = label[i] - label[i - 1]
                    L2 = L2 + loss(deltaP, deltaG)
            Ls1.append(L1)
            Ls2.append(L2) if seqLength > 1 else Ls2.append(nd.zeros(1))

        # number
        n1 = n1 + len(ctx) * batchsize * seqLength
        n2 = n2 + len(ctx) * batchsize * (seqLength - 1)

        # loss
        for i in range(len(loss1)):
            loss1[i] += Ls1[i]
            loss2[i] += Ls2[i]

        # metric, save time
        if UseMetric:
            for pred_batch, label_batch in zip(output_list, label_list):  # for each timestamp
                for t_pred, t_label in zip(pred_batch, label_batch):
                    eval_metric.update(t_label, t_pred)
    nd.waitall()
    loss1 = sum([item.sum().asscalar() for item in loss1])
    loss2 = sum([item.sum().asscalar() for item in loss2])
    validloss = loss1 / n1 + w * loss2 / n2
    MPJPE = eval_metric.get()[-1].sum(axis=0) / 17 if UseMetric else 0

    logger.info("VALID - Epoch:%2d Loss1:%.2e Loss2(%2d):%.2e TotalLoss:%.2e MPJPE:%.1f" %
                (epoch + 1, loss1 / n1, w, loss2 / n2, validloss, MPJPE))


def TestNet_Batch(net, test_data, loss, avg_metric, xyz_metric, mean3d, std3d, config, logger, ctx):
    if not logger:
        assert False, 'require a logger'

    if avg_metric and xyz_metric:
        avg_metric.reset()
        xyz_metric.reset()
    test_data.reset()

    batchsize = config.TEST.batchsize
    seqLength = config.DATASET.seqLength
    nJoints = config.NETWORK.nJoints
    loss1, loss2, n1, n2 = [0] * len(ctx), [0] * len(ctx), 0.000001, 0.000001

    for batch_i, batch in enumerate(test_data):
        data_list = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=1)
        label_list = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=1)

        Ls1, Ls2, output_list = [], [], []
        # forward
        for data, label, cx in zip(data_list, label_list, ctx):
            initial_state = [nd.zeros(shape=(batchsize, config.NETWORK.hidden_dim), ctx=cx) for _ in range(2)]
            start_token = nd.ones(shape=(batchsize, 3 * nJoints), ctx=cx)
            preds = net(data, initial_state, start_token)
            output_list.append(preds)  # pred=[seqLength, 64x48]

            L1, L2 = 0, 0
            for pd, lb in zip(preds, label):
                L1 = L1 + loss(pd, lb)
            if seqLength > 1:
                for i in range(1, seqLength):
                    deltaP = preds[i] - preds[i - 1]
                    deltaG = label[i] - label[i - 1]
                    L2 = L2 + loss(deltaP, deltaG)
            Ls1.append(L1)
            Ls2.append(L2) if seqLength > 1 else Ls2.append(nd.zeros(1))

        # number
        n1 = n1 + len(ctx) * batchsize * seqLength
        n2 = n2 + len(ctx) * batchsize * (seqLength - 1)

        # loss
        for i in range(len(loss1)):
            loss1[i] += Ls1[i]
            loss2[i] += Ls2[i]

        # metric, last frame
        for label_batch, pred_batch in zip(label_list, output_list):
            avg_metric.update(label_batch[-1], pred_batch[-1])
            xyz_metric.update(label_batch[-1], pred_batch[-1])

    # record
    MPJPE = avg_metric.get()[-1].sum(axis=0) / 17
    jntErr = avg_metric.get()[-1]
    xyzErr = xyz_metric.get()[-1]
    loss1 = sum([item.sum().asscalar() for item in loss1])
    loss2 = sum([item.sum().asscalar() for item in loss2])

    return [[n1, n2], [loss1, loss2], MPJPE, xyzErr, jntErr]
