import os

import dataProcess
import logging
from torch.utils.tensorboard import SummaryWriter


class Modeltest(object):
    def __init__(self):
        pass
    @classmethod
    def val(cls,fold):
        from model import EmoTransformer
        from train import config
        from dataProcess import Dataprocess
        import torch
        import os
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            level=logging.INFO,
                            filename=config['LOG_pth'],
                            filemode='a')
        trans = EmoTransformer(input=config['T_input_dim'], nhead=config['T_head_num'],
                               num_layers=config['T_block_num'], batch_first=config['T_bs_first'],
                               output_dim=config['T_output_dim'])
        trans.cuda()
        trans.eval()
        acc, acc_hat, score, total = 0, 0, 0, 0
        checkpoints_pth = os.listdir(config['checkpoint_pth'])
        writer_acc_val = SummaryWriter(f'./tb/acc/val/')
        for cp in checkpoints_pth:
            if cp.startswith(f'{fold}test_{11-fold}val_'):
                checkpoint = torch.load(os.path.join(config['checkpoint_pth'],cp))
                trans.load_state_dict(checkpoint['state_dict'])

                label_val, feature_val, lms3d_val, seqs_val = Dataprocess.loadSingleFold(11-fold, True)
                val_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature_val, lms3d_val, label_val,False,True)
                pred_lst,label_lst,belong_lst=[], [], []
                for input, target, attribution in val_dataloader:
                    label_lst.append(target)
                    pred = trans(input, config['T_masked'])
                    pred_lst.append(pred)
                    belong_lst.append(attribution)
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                    writer_acc_val.add_scalar('val acc', score / total * 100, checkpoint['epoch'])
                vote_acc = dataProcess.Utils.vote(pred_lst,belong_lst,label_lst)
                if acc < score / total :
                    acc = score / total
                    logging.info(f'acc highest:{acc * 100}% when vote is {vote_acc*100}%, epoch is {checkpoint["epoch"]}')
                if acc_hat<vote_acc:
                    acc_hat = vote_acc
                    logging.info(f'vote highest:{acc_hat*100}% when acc is {score / total*100}%, epoch is {checkpoint["epoch"]}')
                print('epoch:',checkpoint['epoch'],(score/total).item()," ",vote_acc.item())
    @classmethod
    def test(cls,epoch,fold):
        from model import EmoTransformer
        from train import config
        from dataProcess import Dataprocess
        import torch
        import os
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            level=logging.INFO,
                            filename=config['LOG_pth'],
                            filemode='a')
        trans = EmoTransformer(input=config['T_input_dim'], nhead=config['T_head_num'],
                               num_layers=config['T_block_num'], batch_first=config['T_bs_first'],
                               output_dim=config['T_output_dim'])
        trans.cuda()
        trans.eval()
        acc, acc_hat, score, total = 0, 0, 0, 0
        checkpoints_pth = os.listdir(config['checkpoint_pth'])
        for cp in checkpoints_pth:
            if cp.startswith(f'{fold}test_{11 - fold}val_'):
                checkpoint = torch.load(os.path.join(config['checkpoint_pth'], cp))
                if checkpoint['epoch'] == 0:
                    trans.load_state_dict(checkpoint['state_dict'])

                    label_val, feature_val, lms3d_val, seqs_val = Dataprocess.loadSingleFold(fold, True)
                    val_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'], feature_val, lms3d_val,
                                                                      label_val, False, True)
                    pred_lst, label_lst, belong_lst = [], [], []
                    for input, target, attribution in val_dataloader:
                        label_lst.append(target)
                        pred = trans(input, config['T_masked'])
                        pred_lst.append(pred)
                        belong_lst.append(attribution)
                        idx_pred = torch.topk(pred, 1, dim=1)[1]
                        rs = idx_pred.eq(target.reshape(-1, 1))
                        score += rs.view(-1).float().sum()
                        total += input.shape[0]
                    vote_acc = dataProcess.Utils.vote(pred_lst, belong_lst, label_lst)
                    if acc < score / total:
                        acc = score / total
                        logging.info(
                            f'acc highest:{acc * 100}% when vote is {vote_acc * 100}%, epoch is {checkpoint["epoch"]}')
                    if acc_hat < vote_acc:
                        acc_hat = vote_acc
                        logging.info(
                            f'vote highest:{acc_hat * 100}% when acc is {score / total * 100}%, epoch is {checkpoint["epoch"]}')
                    print('epoch:', checkpoint['epoch'], (score / total).item(), " ", vote_acc.item())

def deleteStaticDict(pth,epoch):
    import torch
    dict_lst = os.listdir(pth)
    delete_lst = []
    for d in dict_lst:
        dict_pth = os.path.join((pth, d))
        checkpoint = torch.load(dict_pth)
        if checkpoint['epoch'] == epoch:
            print(f'keep dict {dict_pth}')
            pass
        else:
            delete_lst.append(dict_pth)
    for dele in delete_lst:
        os.remove(dele)
if __name__ == '__main__':
    # Modeltest.val(5)
    # Modeltest.test(890,5)
    deleteStaticDict('./',0)