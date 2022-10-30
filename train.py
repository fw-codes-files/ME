import os
import time

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from dataProcess import Dataprocess, LSTMDataSet
from model import LSTMModel
import yaml
from tqdm import trange
from dataProcess import EOS
import logging
import math

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
                    feature[start_idx:start_idx + int(sq)] = (feature[start_idx:start_idx + int(sq)] - f_mean.view(-1,
                                                                                                                   1)) / f_std.view(
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
        acc_lst = [0] * 10
        if fast:
            f_lst = [1, 5, 9]
        else:
            f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for fold in f_lst:
            lstm = LSTMModel(inputDim=config['LSTM_input_dim'], hiddenNum=config['LSTM_hidden_dim'],
                             outputDim=config['LSTM_output_dim'], layerNum=config['LSTM_layerNum'],
                             cell=config['LSTM_cell'], use_cuda=config['use_cuda'],feature_dim=64)
            lstm.cuda()
            optimizer = torch.optim.RMSprop(lstm.parameters(), lr=float(config['learning_rate']), momentum=0.9)
            loss_func = torch.nn.CrossEntropyLoss()
            label, feature, lms3d, seqs, label_test, feature_test, lms3d_test, seqs_test = Dataprocess.dataForLSTM(fold, crop=True)
            # pca_lms3d = Dataprocess.pca(lms3d, seqs, True)
            train_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature, lms3d, label, int(config['Sample_frequency']))
            # pca_lms3d_test = Dataprocess.pca(lms3d_test,seqs_test, False)
            test_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature_test, lms3d_test, label_test, int(config['Sample_frequency']))
            # acc = getAcc(fold)
            acc = 0
            for e in trange(epoch):
                lstm.train()
                score = 0
                total = 0
                # train
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
                    print('\r' + f'fold:{fold} epoch:{e} loss:{loss} acc:{score / total * 100}% ', end='', flush=True)
                    if math.isnan(loss):
                        break
                # eval
                score_t = 0
                total_t = 0
                lstm.eval()
                for test_input, test_label in test_dataloader:
                    pred = lstm(test_input)
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(test_label.reshape(-1, 1))
                    score_t += rs.view(-1).float().sum()
                    total_t += test_input.shape[0]
                print(f'eval acc:{score_t / total_t * 100}%')
                if cut:
                    acc = score_t / total_t
                    acc_lst[fold - 1] = acc
                    logging.info(f'./weights/minibatch_fold_{fold}_epoch_{e}_acc_{acc}_bs_{config["batch_size"]}_lr_{config["learning_rate"]}_ln_{config["LSTM_layerNum"]}_hd_{config["LSTM_hidden_dim"]}_ws_{config["window_size"]}')
                else:
                    if acc < score_t / total_t:
                        acc = score_t / total_t
                        acc_lst[fold - 1] = acc
                        logging.info(f'./weights/minibatch_fold_{fold}_epoch_{e}_acc_{acc}_bs_{config["batch_size"]}_lr_{config["learning_rate"]}_ln_{config["LSTM_layerNum"]}_hd_{config["LSTM_hidden_dim"]}_ws_{config["window_size"]}')
            logging.info(f'------------------------------------fold{fold} ends-----------------------------------------------')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logging.info(f'{len(f_lst)} folds average acc is {sum(acc_lst) / len(f_lst)}')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

class Transformer_traintest():
    def __init__(self):
        pass
    @classmethod
    def train(cls, fast, epoch):
        from model import EmoTransformer
        from torch.utils.tensorboard import SummaryWriter
        acc_lst = [0] * 10
        if fast:
            f_lst = [1, 5, 9]
        else:
            f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for fold in f_lst:
            trans = EmoTransformer(input=204,nhead=config['T_head_num'],num_layers=config['T_block_num'],batch_first=config['T_bs_first'],output_dim=config['T_output_dim'])
            trans.cuda()
            optimizer = torch.optim.Adam(trans.parameters(),lr=1e-6, weight_decay=0.05)
            loss_func = torch.nn.CrossEntropyLoss()
            label, feature, lms3d, seqs, label_test, feature_test, lms3d_test, seqs_test = Dataprocess.dataForLSTM(fold, crop=True)
            train_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature, lms3d, label, int(config['Sample_frequency']))
            test_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature_test, lms3d_test, label_test, int(config['Sample_frequency']))

            acc = 0
            writer_loss = SummaryWriter(f'./tb/loss/{fold}')
            writer_acc_train = SummaryWriter(f'./tb/acc/train/{fold}')
            writer_acc_test = SummaryWriter(f'./tb/acc/test/{fold}')
            for e in range(epoch):
                trans.train()
                score,score_t,total,total_t = 0,0,0,0
                for input, target in train_dataloader:
                    optimizer.zero_grad()
                    pred = trans(input)
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
                    logging.info(f'minibatch_fold_{fold}_epoch_{e}_\ttrainAcc_{score/total*100} %\t_\ttestAcc{acc*100} %\t_bs_{config["batch_size"]}_lr_{config["learning_rate"]}_ln_{6}_hd_{6}_ws_{config["window_size"]}')
            logging.info(f'------------------------------------fold{fold} ends-----------------------------------------------')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logging.info(f'{len(f_lst)} folds average acc is {sum(acc_lst) / len(f_lst)}')
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    @classmethod
    def trainBypackage(cls,fast, epoch):
        from vit_pytorch import ViT
        acc_lst = [0] * 10
        if fast:
            f_lst = [1, 5, 9]
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
                    torch.save(ae.state_dict(),f'./weights/AE_model/AE_model_{e}.pth')
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LSTM train function choice')
    parser.add_argument('-M', default='transformer', type=str, metavar='N',
                        help='s means single and m means minibatch')
    args = parser.parse_args()
    if args.M == 'single':
        LSTM_model_traintest.train_single_vedio()
    elif args.M == 'mini':
        LSTM_model_traintest.train_mini_batch()
    elif args.M == 'attention':
        LSTM_model_traintest.train_justNfolds(cut=False, epoch=1024, fast=False)
    elif args.M == 'transformer':
        Transformer_traintest.train(True,2000)
    elif args.M == 'p_transformer':
        Transformer_traintest.trainBypackage(True,8192)
    elif args.M == 'ae':
        AutoEncoder.train()

if __name__ == '__main__':
    main()
