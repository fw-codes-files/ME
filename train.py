import os
import time

import tqdm
import yaml
config = yaml.safe_load(open('./config.yaml'))
os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_idx']
import numpy as np

import dataProcess

import torch
from dataProcess import Dataprocess
from model import *
from tqdm import trange
import logging
import math
# from mail import send
config = yaml.safe_load(open('./config.yaml'))

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                    level=logging.INFO,
                                    filename=config['LOG_pth'],
                                    filemode='a')
logging.info(config['LOG_CHECK'])

class LSTM_model_traintest(object):
    def __init__(self):
        pass
    @classmethod
    def train_single_vedio(cls):
        for fold in trange(1, 11):  # 分fold
            lstm = LSTMModel(inputDim=config['LSTM_input_dim'], hiddenNum=config['LSTM_hidden_dim'],
                             outputDim=config['LSTM_output_dim'], layerNum=config['LSTM_layerNum'],
                             cell=config['LSTM_cell'], use_cuda=config['use_cuda'])

            lstm.cuda()
            optimizer = torch.optim.RMSprop(lstm.parameters(), lr=float(config['learning_rate']), momentum=0.9)
            loss_func = torch.nn.CrossEntropyLoss()
            lstm.train()
            label, feature, lms3d, seqs, label_test, feature_test, lms3d_test, seqs_test = Dataprocess.dataForLSTM(fold)
            for e in trange(1024):
                start_idx = 0
                rs = 0
                for sq in seqs:
                    f_mean = torch.mean(feature[start_idx:start_idx + int(sq)], dim=1)
                    f_std = torch.std(feature[start_idx:start_idx + int(sq)], dim=1)
                    feature[start_idx:start_idx + int(sq)] = (feature[start_idx:start_idx + int(sq)] - f_mean.view(-1,1)) / f_std.view(
                        -1, 1)
                    l_mean = torch.mean(lms3d[start_idx * 68:(start_idx + int(sq)) * 68], dim=1)
                    l_std = torch.std(lms3d[start_idx * 68:(start_idx + int(sq)) * 68], dim=1)
                    lms3d[start_idx * 68:(start_idx + int(sq)) * 68] = (lms3d[start_idx * 68:(start_idx + int(
                        sq)) * 68] - l_mean.view(-1, 1)) / l_std.view(-1, 1)
                    avideo = torch.cat((feature[start_idx:start_idx + int(sq)],
                                        lms3d[start_idx * 68:(start_idx + int(sq)) * 68].view(-1, 204)), dim=1)
                    avideo_label = label[start_idx:start_idx + int(sq)]
                    start_idx += int(sq)
                    optimizer.zero_grad()
                    pred = lstm(avideo.unsqueeze(0))
                    loss = loss_func(pred, avideo_label[0].unsqueeze(0).long())
                    pred_idx = torch.topk(pred, 1, dim=1)[1]
                    if pred_idx[0][0].data == avideo_label[0].data:
                        rs += 1
                    loss.requires_grad_(True)
                    loss.backward()
                    optimizer.step()
                print('fold:', fold, 'epoch:', e, 'acc:', rs / len(seqs) * 100, '%')
                if rs / len(seqs) == 1.0:
                    # torch.save({
                    #     'epoch': e,
                    #     'state_dict': lstm.state_dict(),
                    #     'acc': rs / len(seqs)
                    # },
                    #     f'./weights/single-video_fold_{fold}_epoch_{e}_acc_{rs / len(seqs)}_lr_{config["learning_rate"]}_ln_{config["LSTM_layerNum"]}_hd_{config["LSTM_hidden_dim"]}.pth')
                    break
            start_idx = 0
            rs = 0
            lstm.eval()
            for sqt in seqs_test:
                f_mean = torch.mean(feature_test[start_idx:start_idx + int(sqt)], dim=1)
                f_std = torch.std(feature_test[start_idx:start_idx + int(sqt)], dim=1)
                feature_test[start_idx:start_idx + int(sqt)] = (feature_test[start_idx:start_idx + int(sqt)] - f_mean.view(
                    -1,
                    1)) / f_std.view(
                    -1, 1)
                l_mean = torch.mean(lms3d_test[start_idx * 68:(start_idx + int(sqt)) * 68], dim=1)
                l_std = torch.std(lms3d_test[start_idx * 68:(start_idx + int(sqt)) * 68], dim=1)
                lms3d_test[start_idx * 68:(start_idx + int(sqt)) * 68] = (lms3d_test[start_idx * 68:(start_idx + int(
                    sqt)) * 68] - l_mean.view(-1, 1)) / l_std.view(-1, 1)
                avideo = torch.cat((feature_test[start_idx:start_idx + int(sqt)],
                                    lms3d_test[start_idx * 68:(start_idx + int(sqt)) * 68].view(-1, 204)), dim=1)
                avideo_label = label_test[start_idx:start_idx + int(sqt)]
                start_idx += int(sqt)
                pred = lstm(avideo.unsqueeze(0))
                pred_idx = torch.topk(pred, 1, dim=1)[1]
                if pred_idx[0][0].data == avideo_label[0].data:
                    rs += 1
            print('eval:', rs / len(seqs_test) * 100, '%')
    @classmethod
    def getAcc(cls,fold):
        folds = []
        acc = 0
        params = os.listdir('./weights/')
        for p in params:
            if p.startswith('mini'):
                folds.append(p)
        for f in folds:
            if f.startswith(f'minibatch_fold_{fold}_epoch'):
                f_lst = f.split('_')
                acc = float(f_lst[6])
        return acc
    @classmethod
    def train_mini_batch(cls):
        acc_lst = [0]*10
        for fold in trange(1, 11):
            lstm = LSTMModel(inputDim=config['LSTM_input_dim'], hiddenNum=config['LSTM_hidden_dim'],
                             outputDim=config['LSTM_output_dim'], layerNum=config['LSTM_layerNum'],
                             cell=config['LSTM_cell'], use_cuda=config['use_cuda'])
            lstm.cuda()
            optimizer = torch.optim.RMSprop(lstm.parameters(), lr=float(config['learning_rate']), momentum=0.9)
            loss_func = torch.nn.CrossEntropyLoss()
            label, feature, lms3d, _, label_test, feature_test, lms3d_test, _ = Dataprocess.dataForLSTM(fold, crop=True)
            train_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'],feature,lms3d,label)
            test_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'],feature_test,lms3d_test,label_test)
            # acc = getAcc(fold)
            acc = 0
            for e in trange(1024):
                lstm.train()
                score = 0
                total = 0
                # train
                for input,target in train_dataloader:
                    optimizer.zero_grad()
                    pred = lstm(input)
                    loss = loss_func(pred,target.long())
                    loss.backward()
                    optimizer.step()
                    # eval train
                    idx_pred = torch.topk(pred,1,dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1,1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                    print('\r' + f'fold:{fold} epoch:{e} loss:{loss} acc:{score/total*100}% ',end='', flush=True)
                    if math.isnan(loss):
                        break
                # eval
                score_t = 0
                total_t = 0
                lstm.eval()
                for test_input,test_label in test_dataloader:
                    pred = lstm(test_input)
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(test_label.reshape(-1, 1))
                    score_t += rs.view(-1).float().sum()
                    total_t += test_input.shape[0]
                print(f'eval acc:{score_t/total_t * 100}%')
                if acc < score_t/total_t:
                    acc = score_t/total_t
                    print(f'find a better model and eval acc:{acc*100}%')
                    acc_lst[fold-1] = acc
                    logging.info(f'./weights/minibatch_fold_{fold}_epoch_{e}_acc_{acc}_bs_{config["batch_size"]}_lr_{config["learning_rate"]}_ln_{config["LSTM_layerNum"]}_hd_{config["LSTM_hidden_dim"]}_ws_{config["window_size"]}')
                    # torch.save({'state_dict': lstm.state_dict(),'EOS':EOS},f'./weights/minibatch_fold_{fold}_epoch_{e}_acc_{acc}_bs_{config["batch_size"]}_lr_{config["learning_rate"]}_ln_{config["LSTM_layerNum"]}_hd_{config["LSTM_hidden_dim"]}_ws_{config["window_size"]}.pth')
            logging.info(f'------------------------------------fold{fold} ends-----------------------------------------------')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logging.info(f'10 folds average acc is {sum(acc_lst)/len(acc_lst)}')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    @classmethod
    def train_justNfolds(cls,cut:bool, epoch:int = 500, fast:bool = True):
        # more flexible than previous methods
        '''
            args:
                cut: if Ture, record accuracy at last epoch; if False, record best accuracy of every fold.
                epoch:
        '''
        from torch.utils.tensorboard import SummaryWriter
        if fast:
            f_lst = config['fast_fold']
        else:
            f_lst = [1, 2, 3, 4, 5]
        for i in range(1,2):
            f_lst = [1, 2, 3, 4, 5]
            f_lst.remove(i)
            lstm = ORILSTM(inputDim=config['LSTM_input_dim'], hiddenNum=config['LSTM_hidden_dim'],
                           outputDim=config['LSTM_output_dim'], layerNum=config['LSTM_layerNum'],
                           cell=config['LSTM_cell'], use_cuda=config['use_cuda'])
            lstm.cuda()
            optimizer = torch.optim.AdamW(lstm.parameters(), lr=float(config['learning_rate']),betas=(0.9, 0.95))
            loss_func = torch.nn.CrossEntropyLoss()
            label, lms3d, feature, seqs = np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
            for c in f_lst:
                train_label_c, train_feature_c, train_lms3d_c, train_seqs_c, _ = Dataprocess.loadSingleFoldOF(c)
                label = np.concatenate((label, train_label_c))
                lms3d = np.concatenate((lms3d, train_lms3d_c))
                feature = np.concatenate((feature, train_feature_c))
                seqs = np.concatenate((seqs, train_seqs_c))
            train_label_c, train_feature_c, train_lms3d_c, train_seqs_c, _ = Dataprocess.loadSingleFoldOF(i, True)
            label = np.concatenate((label, train_label_c))
            lms3d = np.concatenate((lms3d, train_lms3d_c))
            feature = np.concatenate((feature, train_feature_c))
            seqs = np.concatenate((seqs, train_seqs_c))
            # pos_embed = np.concatenate((pos_embed, train_pos_embedd))
            train_dataloader = Dataprocess.VideoNaive(config['window_size'], feature, None, label, False, pos_embed=None)
            lstm.train()
            writer_loss = SummaryWriter(f'./tb/loss/B16_OF_lstm/{i}/')
            writer_acc_train = SummaryWriter(f'./tb/acc/B16_OF_lstm/{i}/')
            for e in trange(epoch):
                score = 0
                total = 0
                for input, target in train_dataloader:
                    optimizer.zero_grad()
                    pred = lstm(input)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    # eval train
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                    print('\r' + f'fold:{i} epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
                    if math.isnan(loss):
                        break
                sacal_loss = loss.detach().cpu()
                writer_loss.add_scalar(f'train loss', sacal_loss, e)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                if (e+1)%1==0:
                    checkpoints = {'model_type': 'lstm',
                                   'epoch': e + 1,
                                   'optim': optimizer.state_dict(),
                                   'state_dict': lstm.state_dict()}
                    torch.save(checkpoints, f'/home/exp-10086/Project/ferdataset/ourFace/lstmRgb/{i}test_{e + 1}.pkl')
            logging.info(f'------------------------------------fold{c} ends-----------------------------------------------')
    @classmethod
    def lstmAB(cls):
        from torch.utils.tensorboard import SummaryWriter
        for i in range(3, 4):
            f_lst = [1, 2, 3, 4, 5]
            f_lst.remove(i)
            lstm = FollowLSTM(inputDim=config['LSTM_input_dim'], hiddenNum=config['LSTM_hidden_dim'],
                           outputDim=config['LSTM_output_dim'], layerNum=config['LSTM_layerNum'],
                           cell=config['LSTM_cell'], use_cuda=config['use_cuda'])
            lstm.cuda()
            lstm.train()
            optimizer = torch.optim.AdamW(lstm.parameters(), lr=float(config['learning_rate']), betas=(0.9, 0.95))
            loss_func = torch.nn.CrossEntropyLoss()
            writer_loss = SummaryWriter(f'./tb/loss/B16_OF_lstm/{i}/')
            writer_acc_train = SummaryWriter(f'./tb/acc/B16_OF_lstm/{i}/')
            train_dataloader = Dataprocess.loadFERModelIntoDataloader(i,'train')
            for e in trange(config['epoch']):
                score = 0
                total = 0
                for input, target,_ in tqdm.tqdm(train_dataloader):
                    optimizer.zero_grad()
                    pred = lstm(input)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    # eval train
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                    print('\r' + f'fold:{i} epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
                    if math.isnan(loss):
                        break
                sacal_loss = loss.detach().cpu()
                writer_loss.add_scalar(f'train loss', sacal_loss, e)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                if (e + 1) % 1 == 0:
                    checkpoints = {'model_type': 'lstm',
                                   'epoch': e + 1,
                                   'optim': optimizer.state_dict(),
                                   'state_dict': lstm.state_dict()}
                    torch.save(checkpoints, f'/home/exp-10086/Project/ferdataset/ourFace/lstmRgb/{i}test_{e + 1}.pkl')
            logging.info(f'------------------------------------fold{i} ends-----------------------------------------------')
class Transformer_traintest():
    def __init__(self):
        pass

    @classmethod
    def train(cls, fast, epoch):
        from model import EmoTransformer
        from torch.utils.tensorboard import SummaryWriter
        acc_lst = [0] * 10
        if fast:
            f_lst = config['fast_fold']
        else:
            f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for fold in f_lst:
            trans = EmoTransformer(input=config['T_input_dim'],nhead=config['T_head_num'],num_layers=config['T_block_num'],batch_first=config['T_bs_first'],output_dim=config['T_output_dim'])
            trans.cuda()
            optimizer = torch.optim.Adam(trans.parameters(),lr=1e-6, weight_decay=0.05)
            loss_func = torch.nn.CrossEntropyLoss()
            # label, feature, lms3d, seqs, label_test, feature_test, lms3d_test,_ = Dataprocess.dataForLSTM(fold, crop=True)
            label, feature, lms3d, seqs = Dataprocess.loadSingleFold(fold, crop=True)
            label_test, feature_test, lms3d_test, seqs_test = Dataprocess.loadSingleFold(fold+1, crop=True)
            train_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature, lms3d, label)
            test_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature_test, lms3d_test, label_test)

            acc = 0
            writer_loss = SummaryWriter(f'./tb/loss/{fold}')
            writer_acc_train = SummaryWriter(f'./tb/acc/train/{fold}')
            writer_acc_test = SummaryWriter(f'./tb/acc/test/{fold}')
            for e in range(epoch):
                trans.train()
                score,score_t,total,total_t = 0,0,0,0
                for input, target in train_dataloader:
                    optimizer.zero_grad()
                    pred = trans(input, config['T_masked'])
                    loss = loss_func(pred,target.long())
                    loss.backward()
                    optimizer.step()
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                sacal_loss = loss.detach().cpu()
                writer_loss.add_scalar('train loss',sacal_loss,e)
                print('\r' + f'fold:{fold} epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
                writer_acc_train.add_scalar('train acc',score / total * 100,e)
                # if e%5 == 0:
                #     checkpoints = {'model_type':'ViT',
                #                    'epoch':e,
                #                    'bs':config["batch_size"],
                #                    'lr':config["learning_rate"],
                #                    'ln':config["T_block_num"],
                #                    'hd':config["T_head_num"],
                #                    'ws':config["window_size"],
                #                    'mask':config["T_masked"],
                #                    'input_dim':config["T_input_dim"],
                #                    'proj_dim':config["T_proj_dim"],
                #                    'forward_dim':config["T_forward_dim"],
                #                    'state_dict':trans.parameters()}
                #     torch.save(checkpoints,f'./{fold}_{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.pkl')
                trans.eval()
                for test_input, test_label in test_dataloader:
                    pred = trans(test_input, config['T_masked'])
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(test_label.reshape(-1, 1))
                    score_t += rs.view(-1).float().sum()
                    total_t += test_input.shape[0]
                writer_acc_test.add_scalar('test acc', score_t / total_t * 100, e)
                if acc < score_t / total_t:
                    acc = score_t / total_t
                    acc_lst[fold - 1] = acc
                    logging.info(f'minibatch_fold_{fold}_epoch_{e}_\ttrainAcc_{score/total*100} %\t_\ttestAcc{acc*100} %\t_bs_{config["batch_size"]}_lr_{config["learning_rate"]}_ln_{config["T_block_num"]}_hd_{config["T_head_num"]}_ws_{config["window_size"]}')
            logging.info(f'------------------------------------fold{fold} ends-----------------------------------------------')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logging.info(f'{len(f_lst)} folds average acc is {sum(acc_lst) / len(f_lst)}')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    @classmethod
    def trainBypackage(cls,fast, epoch):
        from vit_pytorch import ViT
        acc_lst = [0] * 10
        if fast:
            f_lst = config['fast_fold']
        else:
            f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        v = ViT(
            image_size = 204,
            patch_size = 1,
            num_classes = 7,
            dim = 204,
            depth = 6,
            heads = 6,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        v.cuda()
        optimizer = torch.optim.Adam(v.parameters(), lr=1e-6, weight_decay=0.05)
        loss_func = torch.nn.CrossEntropyLoss()
        for fold in f_lst:
            label, feature, lms3d, seqs, label_test, feature_test, lms3d_test, seqs_test = Dataprocess.dataForLSTM(fold,
                                                                                                                   crop=True)
            train_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature, lms3d, label,
                                                                int(config['Sample_frequency']))
            test_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature_test, lms3d_test, label_test,
                                                               int(config['Sample_frequency']))
            for e in range(epoch):
                v.train()
                score = 0
                total = 0
                for input, target in train_dataloader:
                    optimizer.zero_grad()
                    pred = v(input)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                print('\r' + f'fold:{fold} epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
                logging.info(f'fold:{fold} epoch:{e} loss:{loss} acc:{score / total * 100}% ')
                score_t = 0
                total_t = 0
                v.eval()
                for test_input, test_label in test_dataloader:
                    pred = v(test_input)
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(test_label.reshape(-1, 1))
                    score_t += rs.view(-1).float().sum()
                    total_t += test_input.shape[0]
                print(f'eval acc:{score_t / total_t * 100}%')

    @classmethod
    def AEandViT(cls, fast, epoch):
        from model import EmoTransformer
        from torch.utils.tensorboard import SummaryWriter
        acc_lst = [0] * 10
        if fast:
            f_lst = config['fast_fold']
        else:
            f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for fold in f_lst:
            trans = EmoTransformer(input=config['AE_mid_dim'], nhead=config['T_head_num'], num_layers=config['T_block_num'],
                                   batch_first=config['T_bs_first'], output_dim=config['T_output_dim'])
            trans.cuda()
            optimizer = torch.optim.Adam(trans.parameters(), lr=1e-6, weight_decay=0.05)
            loss_func = torch.nn.CrossEntropyLoss()
            label, feature, lms3d, seqs, label_test, feature_test, lms3d_test, seqs_test = Dataprocess.dataForLSTM(fold, crop=True)
            ae_lms,ae_lms_test = Dataprocess.AE_feature(config['AE_pth_epoch'],config['AE_mid_dim'],fold,seqs,seqs_test)
            train_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature, ae_lms, label,
                                                               use_AE=True)
            test_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature_test, ae_lms_test,
                                                               label_test, use_AE=True)

            acc = 0
            writer_loss = SummaryWriter(f'./tb/loss/{fold}')
            writer_acc_train = SummaryWriter(f'./tb/acc/train/{fold}')
            writer_acc_test = SummaryWriter(f'./tb/acc/test/{fold}')
            for e in range(epoch):
                trans.train()
                score, score_t, total, total_t = 0, 0, 0, 0
                for input, target in train_dataloader:
                    optimizer.zero_grad()
                    pred = trans(input)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                sacal_loss = loss.detach().cpu()
                writer_loss.add_scalar('train loss', sacal_loss, e)
                print('\r' + f'fold:{fold} epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                trans.eval()
                for test_input, test_label in test_dataloader:
                    pred = trans(test_input)
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(test_label.reshape(-1, 1))
                    score_t += rs.view(-1).float().sum()
                    total_t += test_input.shape[0]
                writer_acc_test.add_scalar('test acc', score_t / total_t * 100, e)
                if acc < score_t / total_t:
                    acc = score_t / total_t
                    acc_lst[fold - 1] = acc
                    logging.info(
                        f'minibatch_fold_{fold}_epoch_{e}_\ttrainAcc_{score / total * 100} %\t_\ttestAcc{acc * 100} %\t_bs_{config["batch_size"]}_lr_{config["learning_rate"]}_ln_{6}_hd_{6}_ws_{config["window_size"]}')
            logging.info(
                f'------------------------------------fold{fold} ends-----------------------------------------------')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logging.info(f'{len(f_lst)} folds average acc is {sum(acc_lst) / len(f_lst)}')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    @classmethod
    def ViLT(cls,fast,epoch):
        from model import ViLT
        from torch.utils.tensorboard import SummaryWriter
        f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        trans = ViLT(config['ViLT_model_type'], config['ViLT_lms_dim'], config['ViLt_rgb_dim'], config['ViLT_embedding_dim'], config['ViLT_head_num'], config['ViLT_forward_dim'], config['ViLT_block_num'], config['ViLT_bs_first'])
        trans.cuda()
        optimizer = torch.optim.Adam(trans.parameters(), lr=1e-6, weight_decay=0.05)
        loss_func = torch.nn.CrossEntropyLoss()
        f_lst.remove(config['test_fold'])
        f_lst.remove(11 - config['test_fold'])
        label, lms3d, feature, seqs = np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
        for c in f_lst:
            train_label_c, train_feature_c, train_lms3d_c, train_seqs_c = Dataprocess.loadSingleFold(c, True)
            label = np.concatenate((label, train_label_c))
            lms3d = np.concatenate((lms3d, train_lms3d_c))
            feature = np.concatenate((feature, train_feature_c))
            seqs = np.concatenate((seqs, train_seqs_c))

        train_dataloader = Dataprocess.baseline(config['window_size'], feature, lms3d, label, False)
        writer_loss = SummaryWriter(f'./tb/loss/train/')
        writer_acc_train = SummaryWriter(f'./tb/acc/train/')
        for e in range(epoch):
            trans.train()
            score, total = 0, 0
            for input, target in train_dataloader:
                optimizer.zero_grad()
                pred = trans(input, config['T_masked'])
                loss = loss_func(pred, target.long())
                loss.backward()
                optimizer.step()
                idx_pred = torch.topk(pred, 1, dim=1)[1]
                rs = idx_pred.eq(target.reshape(-1, 1))
                score += rs.view(-1).float().sum()
                total += input.shape[0]
            sacal_loss = loss.detach().cpu()
            writer_loss.add_scalar('train loss', sacal_loss, e)
            print('\r' + f'epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
            writer_acc_train.add_scalar('train acc', score / total * 100, e)
            if e % 10 == 0:
                checkpoints = {'model_type': 'ViT',
                               'epoch': e,
                               'bs': config["batch_size"],
                               'lr': config["learning_rate"],
                               'ln': config["T_block_num"],
                               'hd': config["T_head_num"],
                               'ws': config["window_size"],
                               'mask': config["T_masked"],
                               'input_dim': config["T_input_dim"],
                               'proj_dim': config["T_proj_dim"],
                               'forward_dim': config["T_forward_dim"],
                               'state_dict': trans.state_dict()}
                torch.save(checkpoints,
                           f'E:/ViT/{config["test_fold"]}test_{11 - config["test_fold"]}val_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.pkl')
        logging.info(
            f'------------------------------------train ends, train folds:1,2,3,4,7,8,9,10-----------------------------------------------')

    @classmethod
    def voteVit(cls,fast,epoch):
        from model import EmoTransformer
        from torch.utils.tensorboard import SummaryWriter
        acc_lst = [0] * 10
        if fast:
            f_lst = config['fast_fold']
        else:
            f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for fold in f_lst:
            trans = EmoTransformer(input=config['T_input_dim'], nhead=config['T_head_num'],
                                   num_layers=config['T_block_num'], batch_first=config['T_bs_first'],
                                   output_dim=config['T_output_dim'])
            trans.cuda()
            optimizer = torch.optim.Adam(trans.parameters(), lr=float(config['learning_rate']), weight_decay=0.05)
            loss_func = torch.nn.CrossEntropyLoss()
            label, feature, lms3d, seqs, label_test, feature_test, lms3d_test, seqs_test = Dataprocess.dataForLSTM(fold, crop=True)
            train_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature, lms3d, label, use_AE=False, vote=config['vote'])
            test_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature_test, lms3d_test, label_test, use_AE=False, vote=config['vote'])

            acc, acc_hat = 0, 0
            writer_loss = SummaryWriter(f'./tb/loss/{fold}')
            writer_acc_train = SummaryWriter(f'./tb/acc/train/{fold}')
            writer_acc_test = SummaryWriter(f'./tb/acc/test/{fold}')

            for e in range(epoch):
                trans.train()
                score, score_t, total, total_t = 0, 0, 0, 0
                train_collection,train_collection_label, train_group_notes = [],[],[]
                for input, target, attribuion in train_dataloader:
                    train_collection_label.append(target)
                    optimizer.zero_grad()
                    pred = trans(input, att_mask=config['T_masked'])
                    train_collection.append(pred)
                    train_group_notes.append(attribuion)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]

                # acc@one sample
                sacal_loss = loss.detach().cpu()
                writer_loss.add_scalar('train loss', sacal_loss, e)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                accVideo = dataProcess.Utils.vote(train_collection,train_group_notes,train_collection_label)

                trans.eval()
                test_collection, test_collection_label, test_group_notes = [],[],[]
                for test_input, test_label, test_attribution in test_dataloader:
                    test_collection_label.append(test_label)
                    pred = trans(test_input, att_mask=config['T_masked'])
                    test_collection.append(pred)
                    test_group_notes.append(test_attribution)
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(test_label.reshape(-1, 1))
                    score_t += rs.view(-1).float().sum()
                    total_t += test_input.shape[0]
                writer_acc_test.add_scalar('test acc', score_t / total_t * 100, e)
                accVideo_t = dataProcess.Utils.vote(test_collection, test_group_notes, test_collection_label)
                print('\r' + f'epoch{e}, loss{loss}, train sample acc{score / total * 100}%, test sample acc{score_t / total_t * 100}%', end="", flush=True)
                print('\r' + f'epoch{e}, loss{loss}, train sample acc{score / total * 100}%, test sample acc{score_t / total_t * 100}%, train video acc{accVideo * 100}%, test video acc{accVideo_t * 100}%', end="", flush=True)
                if acc < accVideo_t:
                    acc = accVideo_t
                    acc_lst[fold - 1] = acc
                    acc_hat = score_t / total_t
                    logging.info(f'fold_{fold} epoch_{e} SampleTrainAcc_{score / total * 100}% SampleTestAcc_{acc_hat * 100}% VideoTrainAcc_{accVideo * 100}% VideoTestAcc_{acc*100}% bs_{config["batch_size"]} lr_{config["learning_rate"]} ln_{config["T_block_num"]} hd_{config["T_head_num"]} ws_{config["window_size"]} record vote')
                if acc_hat < score_t / total_t:
                    acc_hat = score_t / total_t
                    acc_lst[fold - 1] = acc_hat
                    logging.info(f'fold_{fold} epoch_{e} SampleTrainAcc_{score / total * 100}% SampleTestAcc_{acc_hat * 100}% bs_{config["batch_size"]} lr_{config["learning_rate"]} ln_{config["T_block_num"]} hd_{config["T_head_num"]} ws_{config["window_size"]} record sample')

            logging.info(f'------------------------------------fold{fold} ends-----------------------------------------------')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logging.info(f'{len(f_lst)} folds average acc is {sum(acc_lst) / len(f_lst)}')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    @classmethod
    def train8Folds4ValAndTest(cls, epoch):
        from model import EmoTransformer
        from torch.utils.tensorboard import SummaryWriter
        f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        trans = EmoTransformer(input=config['T_input_dim'], nhead=config['T_head_num'], num_layers=config['T_block_num'], batch_first=config['T_bs_first'], output_dim=config['T_output_dim'])
        trans.cuda()
        optimizer = torch.optim.Adam(trans.parameters(), lr=1e-6, weight_decay=0.05)
        loss_func = torch.nn.CrossEntropyLoss()
        f_lst.remove(config['test_fold'])
        f_lst.remove(11-config['test_fold'])
        label, lms3d, feature, seqs = np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
        for c in f_lst:
            train_label_c, train_feature_c, train_lms3d_c, train_seqs_c = Dataprocess.loadSingleFold(c, True)
            label = np.concatenate((label, train_label_c))
            lms3d = np.concatenate((lms3d, train_lms3d_c))
            feature = np.concatenate((feature, train_feature_c))
            seqs = np.concatenate((seqs, train_seqs_c))

        train_dataloader = Dataprocess.ConvertVideo2Samples(config['window_size'], feature, lms3d, label, False)
        writer_loss = SummaryWriter(f'./tb/loss/train/')
        writer_acc_train = SummaryWriter(f'./tb/acc/train/')
        for e in range(epoch):
            trans.train()
            score, total = 0, 0
            for input, target in train_dataloader:
                optimizer.zero_grad()
                pred = trans(input, config['T_masked'])
                loss = loss_func(pred, target.long())
                loss.backward()
                optimizer.step()
                idx_pred = torch.topk(pred, 1, dim=1)[1]
                rs = idx_pred.eq(target.reshape(-1, 1))
                score += rs.view(-1).float().sum()
                total += input.shape[0]
            sacal_loss = loss.detach().cpu()
            writer_loss.add_scalar('train loss', sacal_loss, e)
            print('\r' + f'epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
            writer_acc_train.add_scalar('train acc', score / total * 100, e)
            if e % 1 == 0:
                checkpoints = {'model_type': 'ViT',
                               'epoch':e,
                               'bs': config["batch_size"],
                               'lr': config["learning_rate"],
                               'ln': config["T_block_num"],
                               'hd': config["T_head_num"],
                               'ws': config["window_size"],
                               'mask': config["T_masked"],
                               'input_dim': config["T_input_dim"],
                               'proj_dim': config["T_proj_dim"],
                               'forward_dim': config["T_forward_dim"],
                               'state_dict': trans.state_dict()}
                torch.save(checkpoints, f'/home/exp-10086/Project/Data/ViT/{config["test_fold"]}test_{11 - config["test_fold"]}val_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.pkl')
        logging.info(f'------------------------------------train ends, train folds:1,2,3,4,7,8,9,10-----------------------------------------------')

    @classmethod
    def linearVote(cls, epoch): # useless
        from model import EmoTransformer
        from torch.utils.tensorboard import SummaryWriter
        f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        trans = EmoTransformer(input=config['T_input_dim'], nhead=config['T_head_num'],
                               num_layers=config['T_block_num'], batch_first=config['T_bs_first'],
                               output_dim=config['T_output_dim'])
        trans.cuda()
        optimizer = torch.optim.Adam(trans.parameters(), lr=1e-6, weight_decay=0.05)
        loss_func = torch.nn.CrossEntropyLoss()
        f_lst.remove(config['test_fold'])
        f_lst.remove(11 - config['test_fold'])
        label, lms3d, feature, seqs = np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
        for c in f_lst:
            train_label_c, train_feature_c, train_lms3d_c, train_seqs_c = Dataprocess.loadSingleFold(c, True)
            label = np.concatenate((label, train_label_c))
            lms3d = np.concatenate((lms3d, train_lms3d_c))
            feature = np.concatenate((feature, train_feature_c))
            seqs = np.concatenate((seqs, train_seqs_c))

        train_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature, lms3d, label, False, True)

        writer_loss = SummaryWriter(f'./tb/loss/train/')
        writer_acc_train = SummaryWriter(f'./tb/acc/train/')
        pred_lst, label_lst, belong_lst = [], [], []
        for e in range(epoch):
            trans.train()
            score, total = 0, 0
            for input, target, attribution in train_dataloader:
                label_lst.append(target)
                optimizer.zero_grad()
                pred = trans(input, config['T_masked'])

                loss = loss_func(pred, target.long())
                loss.backward()
                optimizer.step()
                idx_pred = torch.topk(pred, 1, dim=1)[1]
                rs = idx_pred.eq(target.reshape(-1, 1))
                score += rs.view(-1).float().sum()
                total += input.shape[0]
            sacal_loss = loss.detach().cpu()
            writer_loss.add_scalar('train loss', sacal_loss, e)
            print('\r' + f'epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
            writer_acc_train.add_scalar('train acc', score / total * 100, e)
            if e % 5 == 0:
                checkpoints = {'model_type': 'ViT',
                               'epoch': e,
                               'bs': config["batch_size"],
                               'lr': config["learning_rate"],
                               'ln': config["T_block_num"],
                               'hd': config["T_head_num"],
                               'ws': config["window_size"],
                               'mask': config["T_masked"],
                               'input_dim': config["T_input_dim"],
                               'proj_dim': config["T_proj_dim"],
                               'forward_dim': config["T_forward_dim"],
                               'state_dict': trans.state_dict()}
                torch.save(checkpoints,
                           f'./{config["test_fold"]}test_{11 - config["test_fold"]}val_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.pkl')
            logging.info(
                f'------------------------------------train ends, train folds:1,2,3,4,7,8,9,10-----------------------------------------------')

    @classmethod
    def baseline84(cls, epoch):
        from model import EmoTransformer
        from torch.utils.tensorboard import SummaryWriter
        f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        trans = EmoTransformer(input=config['T_input_dim'], nhead=config['T_head_num'],
                               num_layers=config['T_block_num'], batch_first=config['T_bs_first'],
                               output_dim=config['T_output_dim'])
        trans.cuda()
        optimizer = torch.optim.Adam(trans.parameters(), lr=1e-6, weight_decay=0.05)
        loss_func = torch.nn.CrossEntropyLoss()
        f_lst.remove(config['test_fold'])
        f_lst.remove(11 - config['test_fold'])
        label, lms3d, feature, seqs = np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
        for c in f_lst:
            train_label_c, train_feature_c, train_lms3d_c, train_seqs_c = Dataprocess.loadSingleFold(c, True)
            label = np.concatenate((label, train_label_c))
            lms3d = np.concatenate((lms3d, train_lms3d_c))
            feature = np.concatenate((feature, train_feature_c))
            seqs = np.concatenate((seqs, train_seqs_c))

        train_dataloader = Dataprocess.baseline(config['window_size'], feature, lms3d, label, False)
        writer_loss = SummaryWriter(f'./tb/loss/train/')
        writer_acc_train = SummaryWriter(f'./tb/acc/train/')
        for e in range(epoch):
            trans.train()
            score, total = 0, 0
            for input, target in train_dataloader:
                optimizer.zero_grad()
                pred = trans(input, config['T_masked'])
                loss = loss_func(pred, target.long())
                loss.backward()
                optimizer.step()
                idx_pred = torch.topk(pred, 1, dim=1)[1]
                rs = idx_pred.eq(target.reshape(-1, 1))
                score += rs.view(-1).float().sum()
                total += input.shape[0]
            sacal_loss = loss.detach().cpu()
            writer_loss.add_scalar('train loss', sacal_loss, e)
            print('\r' + f'epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
            writer_acc_train.add_scalar('train acc', score / total * 100, e)
            if e % 10 == 0:
                checkpoints = {'model_type': 'ViT',
                               'epoch': e,
                               'bs': config["batch_size"],
                               'lr': config["learning_rate"],
                               'ln': config["T_block_num"],
                               'hd': config["T_head_num"],
                               'ws': config["window_size"],
                               'mask': config["T_masked"],
                               'input_dim': config["T_input_dim"],
                               'proj_dim': config["T_proj_dim"],
                               'forward_dim': config["T_forward_dim"],
                               'state_dict': trans.state_dict()}
                torch.save(checkpoints,
                           f'E:/ViT/{config["test_fold"]}test_{11 - config["test_fold"]}val_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.pkl')
        logging.info(f'------------------------------------train ends, train folds:1,2,3,4,7,8,9,10-----------------------------------------------')
class AutoEncoder():
    def __init__(self):
        pass
    @classmethod
    def train(cls):
        from model import AutoEncoder
        from torch.utils.tensorboard import SummaryWriter
        writer_loss_train = SummaryWriter(config['AE_loss_train'])
        writer_loss_test = SummaryWriter(config['AE_loss_train'])
        '''
            use mlp to obtain deep feature distribution on expression
        '''
        # all CK+ dataset can be used for training and testing, AE is unsupervision
        ae = AutoEncoder()
        ae.cuda()
        optimizer = torch.optim.SGD(ae.parameters(),lr=1e-6,momentum=0.9)
        loss_func = torch.nn.MSELoss()
        train_dataloader,test_dataloader = Dataprocess.AEdataload()
        for e in trange(1,10001):
            ae.train()
            for input,_ in train_dataloader:
                optimizer.zero_grad()
                output = ae(input)
                loss = loss_func(output,input)
                loss.backward()
                optimizer.step()
                print('train loss:',loss)
            writer_loss_train.add_scalar('AE train loss',loss.item(),e)
            if e<5000:
                pass
            else:
                if e%1000 ==0 :
                    torch.save(ae.state_dict(),f'./weights/AE_model/AE_model_{e}_mid_{16}.pth')
            ae.eval()
            for input,_ in test_dataloader:
                output = ae(input)
                if e == 9999:
                    last = (output[-5],input[-5])
                test_loss = loss_func(output, input)
                print('test_loss:',test_loss)
            writer_loss_test.add_scalar('AE test loss', loss.item(), e)
        import open3d as o3d
        pcd_s = o3d.geometry.PointCloud()
        pcd_pc = o3d.geometry.PointCloud()
        cl0 = np.zeros((68,3))
        cl0[:,0] = 1
        cl1 = np.zeros((68,3))
        cl1[:,1] = 1
        pcd_s.points = o3d.utility.Vector3dVector(last[0].reshape((68,3)).cpu().detach().numpy())
        pcd_s.colors = o3d.utility.Vector3dVector(cl0)
        pcd_pc.points = o3d.utility.Vector3dVector(last[1].reshape((68,3)).cpu().detach().numpy())
        pcd_pc.colors = o3d.utility.Vector3dVector(cl1)
        o3d.visualization.draw_geometries([pcd_s, pcd_pc])
        return
class Two_tsm_train():
    def __init__(self):
        pass
    @classmethod
    def train(cls, epoch):
        from model import MultiEmoTransformer
        from torch.utils.tensorboard import SummaryWriter
        for i in range(1,11):
            f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            trans = MultiEmoTransformer(lms3dpro = config['T_lms3d_dim'], rgbpro = config['T_rgb_dim'], input=config['T_input_dim'], nhead=config['T_head_num'], num_layers=config['T_block_num'], batch_first=config['T_bs_first'], output_dim=config['T_output_dim'])
            trans.cuda()
            optimizer = torch.optim.Adam(trans.parameters(), lr=1e-6, weight_decay=0.05)
            loss_func = torch.nn.CrossEntropyLoss()
            f_lst.remove(i)
            label, lms3d, feature, seqs = np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
            for c in f_lst:
                train_label_c, train_feature_c, train_lms3d_c, train_seqs_c = Dataprocess.loadSingleFold(c, True)
                label = np.concatenate((label, train_label_c))
                lms3d = np.concatenate((lms3d, train_lms3d_c))
                feature = np.concatenate((feature, train_feature_c))
                seqs = np.concatenate((seqs, train_seqs_c))
            train_dataloader = Dataprocess.ConvertVideo2Samples(config['window_size'], feature, lms3d, label, False)
            writer_loss = SummaryWriter(f'./tb/loss/train/')
            writer_acc_train = SummaryWriter(f'./tb/acc/train/')
            for e in range(1,epoch+1):
                trans.train()
                score, total = 0, 0
                for input, target in train_dataloader:
                    optimizer.zero_grad()
                    pred = trans(input[:,:,512:],input[:,:,:512], config['T_masked'])
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                sacal_loss = loss.detach().cpu()
                writer_loss.add_scalar('train loss', sacal_loss, e)
                print('\r' + f'epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                if e % 5 == 0:
                    checkpoints = {'model_type': 'ViT',
                                   'epoch': e,
                                   'bs': config["batch_size"],
                                   'lr': config["learning_rate"],
                                   'ln': config["T_block_num"],
                                   'hd': config["T_head_num"],
                                   'ws': config["window_size"],
                                   'mask': config["T_masked"],
                                   'input_dim': config["T_input_dim"],
                                   'proj_dim': config["T_proj_dim"],
                                   'forward_dim': config["T_forward_dim"],
                                   'state_dict': trans.state_dict()}
                    torch.save(checkpoints, f'/media/exp-10086/Elements SE/ViT/{i}test_{e}.pkl')
            logging.info( f'------------------------------------train ends, train folds:{f_lst}-----------------------------------------------')
            # send('2080ti train result', f'{config["LOG_CHECK"]}')
    @classmethod
    def pre2train(cls, epoch):
        from model import MAEEncoder
        from torch.utils.tensorboard import SummaryWriter
        import timm.optim.optim_factory as optim_factory
        from dataProcess import Utils

        loss_func = torch.nn.CrossEntropyLoss()
        for i in range(1,11):
            f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            trans = MAEEncoder(embed_dim=config['T_proj_dim'], depth=config['T_block_num'], num_heads=config['T_head_num'])
            checkpoint = f'/home/exp-10086/Project/mae-main/output_dir0/checkpoint-399.pth'
            state_dict = torch.load(checkpoint)
            trans = Utils.loadKeys(trans, state_dict['model'])
            trans.cuda()
            trans.train()
            param_groups = optim_factory.add_weight_decay(trans, 0.05)
            optimizer = torch.optim.AdamW(param_groups, lr=1e-3, betas=(0.9, 0.95))
            f_lst.remove(i)
            # f_lst.remove(11 - config['test_fold'])
            label, lms3d, feature, seqs = np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
            for c in f_lst:
                train_label_c, train_feature_c, train_lms3d_c, train_seqs_c = Dataprocess.loadSingleFold(c, True)
                label = np.concatenate((label, train_label_c))
                lms3d = np.concatenate((lms3d, train_lms3d_c))
                feature = np.concatenate((feature, train_feature_c))
                seqs = np.concatenate((seqs, train_seqs_c))
            train_dataloader = Dataprocess.ConvertVideo2Samples100Votes(config['window_size'], feature, lms3d, label, False)
            writer_loss = SummaryWriter(f'./tb/loss/pretrain/{i}/')
            writer_acc_train = SummaryWriter(f'./tb/acc/pretrain/{i}/')
            for e in range(5000):
                score, total = 0, 0
                for input, target in train_dataloader:
                    optimizer.zero_grad()
                    pred = trans(input)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                sacal_loss = loss.detach().cpu()
                writer_loss.add_scalar('train loss', sacal_loss, e)
                print('\r' + f'epoch:{e} loss:{loss} acc:{score / total * 100}% lr:{optimizer.param_groups[0]["lr"]}', end='', flush=True)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                if (e+1) % 5 == 0:
                    checkpoints = {'model_type': 'ViT',
                                   'epoch': e+1,
                                   'optimizer': optimizer.state_dict(),
                                   'state_dict': trans.state_dict()}
                    # torch.save(checkpoints,f'/home/exp-10086/Project/Data/ViT/{config["test_fold"]}test_{11 - config["test_fold"]}val_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.pkl')
                    torch.save(checkpoints,f'/home/exp-10086/Project/Data/ViT/{i}test_{e}.pkl')
            logging.info(f'------------------------------------train ends, train folds:{f_lst}-----------------------------------------------')
    @classmethod
    def B16Vit(cls, epoch):
        from model import NormalB16ViT
        from torch.utils.tensorboard import SummaryWriter
        from numpy import load
        from dataProcess import Utils
        # npz = load('./weights/imagenet21k_ViT-B_16.npz')
        loss_func = torch.nn.CrossEntropyLoss()
        pos_encoding = Utils.SinusoidalEncoding(270, config['T_proj_dim'])
        for i in range(1, 2):
            f_lst = [1, 2, 3, 4, 5]
            nv = NormalB16ViT(None)
            nv.cuda()
            nv.train()
            optimizer = torch.optim.AdamW(nv.parameters(), lr=1e-3, betas=(0.9, 0.95))
            f_lst.remove(i)
            # f_lst.remove(11 - config['test_fold'])
            label, lms3d, feature, seqs, = np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
            for c in f_lst:
                train_label_c, train_feature_c, train_lms3d_c, train_seqs_c, _ = Dataprocess.loadSingleFoldOF(c)
                label = np.concatenate((label, train_label_c))
                lms3d = np.concatenate((lms3d, train_lms3d_c))
                feature = np.concatenate((feature, train_feature_c))
                seqs = np.concatenate((seqs, train_seqs_c))
                # pos_embed = np.concatenate((pos_embed, train_pos_embedd))
            # 第5个fold中再取出来半fold，这样相当于10fold
            train_label_c, train_feature_c, train_lms3d_c, train_seqs_c, _ = Dataprocess.loadSingleFoldOF(i,True)
            label = np.concatenate((label, train_label_c))
            lms3d = np.concatenate((lms3d, train_lms3d_c))
            feature = np.concatenate((feature, train_feature_c))
            seqs = np.concatenate((seqs, train_seqs_c))
            # pos_embed = np.concatenate((pos_embed, train_pos_embedd))
            train_dataloader = Dataprocess.VideoNaive(config['window_size'], feature, lms3d, label, False, None)
            writer_loss = SummaryWriter(f'./tb/loss/B16_OF/{i}/')
            writer_acc_train = SummaryWriter(f'./tb/acc/B16_OF/{i}/')
            for e in range(epoch):
                score, total = 0, 0
                # checkpoint = torch.load(os.path.join(config['checkpoint_pth'], f'{i}test_200.pkl'))
                # nv.load_state_dict(checkpoint['state_dict'])
                # optimizer.load_state_dict(checkpoint['optim'])
                for input, target in train_dataloader:
                    optimizer.zero_grad()
                    pred = nv(input, pos_encoding)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                    sacal_loss = loss.detach().cpu()
                writer_loss.add_scalar(f'train loss', sacal_loss, e)
                print(
                    '\r' + f'fold {i} epoch:{e} loss:{loss} acc:{score / total * 100}% lr:{optimizer.param_groups[0]["lr"]} ',
                    end='', flush=True)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                if (e+1) % 1 == 0:
                    checkpoints = {'model_type': 'ViT',
                                   'epoch': e + 1,
                                   'optim': optimizer.state_dict(),
                                   'state_dict': nv.state_dict()}
                    torch.save(checkpoints, f'/home/exp-10086/Project/ferdataset/ourFace/vit/{i}test_{e+1}.pkl')
            logging.info(f'------------------------------------train ends, train folds:{f_lst}-----------------------------------------------')
    @classmethod
    def B16Vit_AE(cls, epoch):
        from model import NormalB16ViT
        from torch.utils.tensorboard import SummaryWriter
        from numpy import load
        npz = load('/home/exp-10086/Project/ferdataset/ViT-H_14.npz')
        loss_func = torch.nn.CrossEntropyLoss()

        for i in range(1, 11):
            f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            nv = NormalB16ViT(npz)
            nv.cuda()
            nv.train()
            optimizer = torch.optim.AdamW(nv.parameters(), lr=1e-3, betas=(0.9, 0.95))
            f_lst.remove(i)
            # f_lst.remove(11 - config['test_fold'])
            label,  feature, pos_embed, threeD = np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
            for c in f_lst:
                train_label_c, train_3d_c, train_pos_embedd, train_feature_c = Dataprocess.readAndLoadSingleFold(c)  # finetune之后的rgb特征
                label = np.concatenate((label, train_label_c))
                feature = np.concatenate((feature, train_feature_c))
                pos_embed = np.concatenate((pos_embed, train_pos_embedd))
                threeD = np.concatenate((threeD, train_3d_c))
            train_dataloader = Dataprocess.ConvertVideo2SamlpesConstantSpeed(config['window_size'], feature, threeD, label, False, pos_embed)
            writer_loss = SummaryWriter(f'./tb/loss/c1AE_Rgbmean/{i}/')
            writer_acc_train = SummaryWriter(f'./tb/acc/c1AE_Rgbmean/{i}/')
            for e in range(epoch):
                score, total = 0, 0
                # checkpoint = torch.load(os.path.join(config['checkpoint_pth'], f'{i}test_200.pkl'))
                # nv.load_state_dict(checkpoint['state_dict'])
                # optimizer.load_state_dict(checkpoint['optim'])
                for input, target, postion_embedding in train_dataloader:
                    optimizer.zero_grad()
                    pred = nv(input, postion_embedding)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                # lr_scheduler.step()
                sacal_loss = loss.detach().cpu()
                writer_loss.add_scalar(f'train loss', sacal_loss, e)
                print(
                    '\r' + f'fold {i} epoch:{e} loss:{loss} acc:{score / total * 100}% lr:{optimizer.param_groups[0]["lr"]}',
                    end='', flush=True)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                if (e + 1) % 5 == 0:
                    checkpoints = {'model_type': 'ViT',
                                   'epoch': e + 1,
                                   'optim': optimizer.state_dict(),
                                   'state_dict': nv.state_dict()}
                    torch.save(checkpoints, f'/home/exp-10086/Project/ferdataset/c1AE_Rgbmean/{i}test_{e + 1}.pkl')
            logging.info(f'------------------------------------train ends, train folds:{f_lst}-----------------------------------------------')
    @classmethod
    def B16Vit_NoMidData(cls, epoch):
        from model import B16ViT_AB
        from torch.utils.tensorboard import SummaryWriter
        from dataProcess import Utils

        loss_func = torch.nn.CrossEntropyLoss()
        pos_encoding = Utils.SinusoidalEncoding(config['window_size'], config['T_proj_dim'])
        for i in range(3, 6): # 用几个fold进行训练
            nv = B16ViT_AB(None)
            nv.cuda()
            nv.train()
            # ckpt = torch.load(os.path.join(config['checkpoint_pth'],f'{i}test_100.pkl'))
            # nv.load_state_dict(ckpt['state_dict'])
            optimizer = torch.optim.AdamW(nv.parameters(), lr=1e-3, betas=(0.9, 0.95))
            # optimizer.load_state_dict(ckpt['optim'])
            # 开始准备数据
            train_dataloader = Dataprocess.loadFERModelIntoDataloader(i,'train')
            # 开始记录指标
            writer_loss = SummaryWriter(f'./tb/loss/vit_rgb_lms_whole/{i}/')
            writer_acc_train = SummaryWriter(f'./tb/acc/vit_rgb_lms_whole/{i}/')
            # 开始训练
            for e in range(1,101):
                score, total = 0, 0
                for input, target,_, lms in tqdm.tqdm(train_dataloader):
                    optimizer.zero_grad()
                    pred = nv(lms.float(), input.float(), pos_encoding)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    #
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                    #
                writer_loss.add_scalar(f'train loss', loss.detach().cpu(), e)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                print('train acc:',score / total * 100, 'epoch', e, f'[{e}/{len(train_dataloader)}]')
                checkpoints = {'model_type': 'ViT',
                               'optim': optimizer.state_dict(),
                               'state_dict': nv.state_dict()}
                torch.save(checkpoints, f'/home/exp-10086/Project/ferdataset/ourFace/vit_rgb_lms_whole/{i}test_{e}.pkl')
    @classmethod
    def B16_AB(cls):
        from torch.utils.tensorboard import SummaryWriter
        from dataProcess import Utils
        loss_func = torch.nn.CrossEntropyLoss()
        pos_encoding = Utils.SinusoidalEncoding(270, config['T_input_dim'])
        for i in range(5, 6):
            nv = B16ViT_AB(None)
            # checkpoint = torch.load(os.path.join(config['checkpoint_pth'], f'{i}test_37.pkl')) # path is needed
            # B16_state = checkpoint['state_dict']
            # model_state_dict = nv.state_dict()
            # for Bk in B16_state:
            #     if Bk.startswith('alpha') or Bk.startswith('beta'):
            #         continue
            #     else:
            #         model_state_dict[Bk] = B16_state[Bk]
            nv.cuda()
            nv.train()
            optimizer = torch.optim.AdamW(nv.parameters(), lr=1e-3, betas=(0.9, 0.95))
            train_dataloader = Dataprocess.loadFERModelIntoDataloader(i, 'train')
            writer_loss = SummaryWriter(f'./tb/loss/B16_rgb_lms_whole_AB_HSE/{i}')
            writer_acc_train = SummaryWriter(f'./tb/acc/B16_rgb_lms_whole_AB_HSE/{i}/')
            for e in range(config['epoch']):
                score, total = 0, 0
                for input, target,_,lms in tqdm.tqdm(train_dataloader, desc=f'{e}'):
                    optimizer.zero_grad()
                    pred = nv(lms.float(),input.float(), pos_encoding)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                    sacal_loss = loss.detach().cpu()
                writer_loss.add_scalar(f'train loss', sacal_loss, e)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                print('\r' + f'fold {i} epoch:{e} loss:{loss} acc:{score / total * 100}% lr:{optimizer.param_groups[0]["lr"]} ',end='', flush=True)
                if (e + 1) % 1 == 0:
                    checkpoints = {'model_type': 'ViT',
                                   'epoch': e + 1,
                                   'optim': optimizer.state_dict(),
                                   'state_dict': nv.state_dict()}
                    torch.save(checkpoints, f'/home/exp-10086/Project/ferdataset/ourFace/B16_rgb_lms_whole_AB_HSE/{i}test_{e + 1}.pkl')
            logging.info(f'------------------------------------train ends, train folds:{i}-----------------------------------------------')
    @classmethod
    def B16_AFEW(cls):
        from model import NormalB16ViT
        from torch.utils.tensorboard import SummaryWriter
        from dataProcess import Utils

        loss_func = torch.nn.CrossEntropyLoss()
        pos_encoding = Utils.SinusoidalEncoding(151, config['T_proj_dim'])
        nv = NormalB16ViT(None)
        # b16_state = torch.load('/home/exp-10086/Project/Data/afew_whole_lms/051.pkl')
        # nv.load_state_dict(b16_state['state_dict'])
        nv.cuda()
        nv.train()
        optimizer = torch.optim.AdamW(nv.parameters(), lr=3e-6, betas=(0.9, 0.95))
        # optimizer.load_state_dict(b16_state['optim'])
        # 开始准备数据
        train_dataloader = Dataprocess.loadAFEWDataset('train')
        # 开始记录指标
        writer_loss = SummaryWriter(f'./tb/loss/B16_AFEW_lms/train/')
        writer_acc_train = SummaryWriter(f'./tb/acc/B16_AFEW_lms/train/')
        # 开始训练
        for e in range(1, 201):
            score, total = 0, 0
            for input, target in tqdm.tqdm(train_dataloader):
                optimizer.zero_grad()
                pred = nv(input.float(), pos_encoding)
                loss = loss_func(pred, target.long())
                loss.backward()
                optimizer.step()
                #
                idx_pred = torch.topk(pred, 1, dim=1)[1]
                rs = idx_pred.eq(target.reshape(-1, 1))
                score += rs.view(-1).float().sum()
                total += input.shape[0]
                #
            if (e + 1) % 1 == 0:
                writer_loss.add_scalar(f'train loss', loss.detach().cpu(), e)
                writer_acc_train.add_scalar('train acc', score / total * 100, e)
                print('train acc:', score / total * 100, 'epoch', e, f'[{e + 1}/{len(train_dataloader)}]')
                checkpoints = {'model_type': 'ViT',
                               'optim': optimizer.state_dict(),
                               'state_dict': nv.state_dict()}
                torch.save(checkpoints, f'/home/exp-10086/Project/Data/afew_whole_lms/0{e}.pkl')
    @classmethod
    def B16_CK(cls):
        from model import NormalB16ViT
        from torch.utils.tensorboard import SummaryWriter
        from dataProcess import Utils

        loss_func = torch.nn.CrossEntropyLoss()
        pos_encoding = Utils.SinusoidalEncoding(71, config['T_proj_dim'])
        nv = NormalB16ViT(None)
        # b16_state = torch.load('/home/exp-10086/Project/Data/afew_whole_lms/051.pkl')
        # nv.load_state_dict(b16_state['state_dict'])
        nv.cuda()
        nv.train()
        optimizer = torch.optim.AdamW(nv.parameters(), lr=3e-6, betas=(0.9, 0.95))
        # optimizer.load_state_dict(b16_state['optim'])
        for fold in range(1,2):
            # 开始准备数据
            train_dataloader = Dataprocess.loadACKDataset(fold,'train')
            # 开始记录指标
            writer_loss = SummaryWriter(f'./tb/loss/B16_CK_lms/train/{fold}')
            writer_acc_train = SummaryWriter(f'./tb/acc/B16_CK_lms/train/{fold}')
            # 开始训练
            for e in range(1, 301):
                score, total = 0, 0
                for input, target, _ in tqdm.tqdm(train_dataloader):
                    optimizer.zero_grad()
                    pred = nv(input.float(), pos_encoding)
                    loss = loss_func(pred, target.long())
                    loss.backward()
                    optimizer.step()
                    #
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                    #
                if (e + 1) % 1 == 0:
                    writer_loss.add_scalar(f'train loss', loss.detach().cpu(), e)
                    writer_acc_train.add_scalar('train acc', score / total * 100, e)
                    print('train acc:', score / total * 100, 'epoch', e, f'[{e + 1}/{len(train_dataloader)}]')
                    checkpoints = {'model_type': 'ViT',
                                   'optim': optimizer.state_dict(),
                                   'state_dict': nv.state_dict()}
                    torch.save(checkpoints, f'/home/exp-10086/Project/Data/ck_whole_lms/{fold}_0{e}.pkl')
def main():
    import argparse
    parser = argparse.ArgumentParser(description='train function choice')
    parser.add_argument('-M', default='B16_AB', type=str, metavar='N',
                        help='s means single and m means minibatch')
    args = parser.parse_args()
    if __name__ == '__main__':
        if args.M == 'single': # lstm train with bs=1
            LSTM_model_traintest.train_single_vedio()
        elif args.M == 'mini': # lstm train with mini bs
            LSTM_model_traintest.train_mini_batch()
        elif args.M == 'attention': # lstm with attention
            LSTM_model_traintest.train_justNfolds(cut=False, epoch=config['epoch'], fast=False)
        elif args.M == 'transformer': # origin vit
            Transformer_traintest.train(True, config['epoch'])
        elif args.M == 'p_transformer': # packaged vit
            Transformer_traintest.trainBypackage(True,config['epoch'])
        elif args.M == 'ae':
            AutoEncoder.train()
        elif args.M == 'aevit':
            Transformer_traintest.AEandViT(True, config['epoch']) # ae dim is related to tsm forward dim
        elif args.M == 'vilt':
            Transformer_traintest.ViLT(True, config['epoch'])
        elif args.M == 'voteVit': # train and test in one epoch and vote
            Transformer_traintest.voteVit(True, config['epoch'])
        elif args.M == '84': # set val dateset and test data set
            Transformer_traintest.train8Folds4ValAndTest(config['epoch'])
        elif args.M == 'linearVote': # model changed vote into linear
            Transformer_traintest.linearVote(config['epoch'])
        elif args.M == 'baseline84': # baseline
            Transformer_traintest.baseline84((config['epoch']))
        elif args.M == '2m84':
            Two_tsm_train.train(config['epoch'])
        elif args.M == 'pretrain':
            Two_tsm_train.pre2train(config['epoch'])
        elif args.M == 'B16':
            Two_tsm_train.B16Vit(config['epoch'])
        elif args.M == 'B16_AE':
            Two_tsm_train.B16Vit_AE(config['epoch'])
        elif args.M == 'B16_check':
            Two_tsm_train.B16Vit_NoMidData(config['epoch'])
        elif args.M == 'AB':
            LSTM_model_traintest.lstmAB()
        elif args.M == 'B16_AB':
            Two_tsm_train.B16_AB()
        elif args.M == 'B16_AFEW':
            Two_tsm_train.B16_AFEW()
        elif args.M == 'B16_CK':
            Two_tsm_train.B16_CK()
if __name__ == '__main__':
    main()
