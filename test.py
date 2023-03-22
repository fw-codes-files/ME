import os

import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch.nn

import dataProcess
import logging
from torch.utils.tensorboard import SummaryWriter

class Modeltest(object):
    softmax = torch.nn.Softmax(dim=1)
    def __init__(self):
        pass
    @classmethod
    def val(cls,fold):
        from model import EmoTransformer,MultiEmoTransformer
        from train import config
        from dataProcess import Dataprocess
        import torch
        import os
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            level=logging.INFO,
                            filename=config['LOG_pth'],
                            filemode='a')
        # trans = EmoTransformer(input=config['T_input_dim'], nhead=config['T_head_num'],
                            #    num_layers=config['T_block_num'], batch_first=config['T_bs_first'],
                            #    output_dim=config['T_output_dim'])
        trans = MultiEmoTransformer(lms3dpro = config['T_lms3d_dim'], rgbpro = config['T_rgb_dim'], input=config['T_input_dim'], nhead=config['T_head_num'], num_layers=config['T_block_num'], batch_first=config['T_bs_first'], output_dim=config['T_output_dim'])
        trans.cuda()
        trans.eval()
        checkpoints_pth = os.listdir(config['checkpoint_pth'])
        writer_acc_val = SummaryWriter(f'./tb/acc/val/')
        acc, acc_hat = 0, 0
        for cp in checkpoints_pth:
            if cp.startswith(f'{fold}test_{11-fold}val_'):
                score, total = 0, 0
                checkpoint = torch.load(os.path.join(config['checkpoint_pth'],cp))
                trans.load_state_dict(checkpoint['state_dict'])
                label_val, feature_val, lms3d_val, seqs_val = Dataprocess.loadSingleFold(fold, True)
                val_dataloader = Dataprocess.ConvertVideo2Samples(config['window_size'], feature_val, lms3d_val, label_val,True) # variables in memory
                pred_lst,label_lst,belong_lst=[], [], []
                for input, target, attribution in val_dataloader:
                    label_lst.append(target)
                    with torch.no_grad():
                        # pred = trans(input, config['T_masked'])
                        pred = trans(input[:,:,512:],input[:,:,:512], config['T_masked'])
                    pred = cls.softmax(pred)
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
                print('epoch:',checkpoint['epoch'],(score/total).item()," ",vote_acc)

    @classmethod
    def test(cls,epoch,fold):
        from model import EmoTransformer,MultiEmoTransformer
        from train import config
        from dataProcess import Dataprocess
        import torch
        import os
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            level=logging.INFO,
                            filename=config['LOG_pth'],
                            filemode='a')
        # trans = EmoTransformer(input=config['T_input_dim'], nhead=config['T_head_num'],
                            #    num_layers=config['T_block_num'], batch_first=config['T_bs_first'],
                            #    output_dim=config['T_output_dim'])
        trans = MultiEmoTransformer(lms3dpro = config['T_lms3d_dim'], rgbpro = config['T_rgb_dim'], input=config['T_input_dim'], nhead=config['T_head_num'], num_layers=config['T_block_num'], batch_first=config['T_bs_first'], output_dim=config['T_output_dim'])
        trans.cuda()
        trans.eval()
        checkpoints_pth = os.listdir(config['checkpoint_pth'])
        acc, acc_hat = 0, 0
        for cp in checkpoints_pth:
            if cp.startswith(f'{fold}test_{11 - fold}val_'):
                score, total = 0, 0
                checkpoint = torch.load(os.path.join(config['checkpoint_pth'], cp))
                if checkpoint['epoch'] == epoch:
                    trans.load_state_dict(checkpoint['state_dict'])

                    label_val, feature_val, lms3d_val, seqs_val = Dataprocess.loadSingleFold(11-fold, True)
                    val_dataloader = Dataprocess.ConvertVideo2Samples(config['window_size'], feature_val, lms3d_val, label_val, True)
                    pred_lst, label_lst, belong_lst = [], [], []
                    for input, target, attribution in val_dataloader:
                        label_lst.append(target)
                        with torch.no_grad():
                            # pred = trans(input, config['T_masked'])
                            pred = trans(input[:,:,512:],input[:,:,:512], config['T_masked'])
                        pred = cls.softmax(pred)
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
                    print('epoch:', checkpoint['epoch'], (score / total).item(), " ", vote_acc)

    @classmethod
    def baselineVal(cls, fold):
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
        checkpoints_pth = os.listdir('E:/ViT/')
        writer_acc_val = SummaryWriter(f'./tb/acc/val/')
        acc, acc_hat = 0, 0
        for cp in checkpoints_pth:
            if cp.startswith(f'{fold}test_{11 - fold}val_'):
                score, total = 0, 0
                checkpoint = torch.load(os.path.join('E:/ViT/', cp))
                trans.load_state_dict(checkpoint['state_dict'])
                label_val, feature_val, lms3d_val, seqs_val = Dataprocess.loadSingleFold(fold, True)
                val_dataloader = Dataprocess.baseline(71, feature_val, lms3d_val, label_val, True)
                pred_lst, label_lst, belong_lst = [], [], []
                for input, target, attribution in val_dataloader:
                    label_lst.append(target)
                    with torch.no_grad():
                        pred = trans(input, config['T_masked'])
                    pred = cls.softmax(pred)
                    pred_lst.append(pred)
                    belong_lst.append(attribution)
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score += rs.view(-1).float().sum()
                    total += input.shape[0]
                    writer_acc_val.add_scalar('val acc', score / total * 100, checkpoint['epoch'])
                vote_acc = dataProcess.Utils.vote(pred_lst, belong_lst, label_lst)
                if acc < score / total:
                    acc = score / total
                    logging.info(
                        f'acc highest:{acc * 100}% when vote is {vote_acc * 100}%, epoch is {checkpoint["epoch"]}')
                if acc_hat < vote_acc:
                    acc_hat = vote_acc
                    logging.info(
                        f'vote highest:{acc_hat * 100}% when acc is {score / total * 100}%, epoch is {checkpoint["epoch"]}')
                print('epoch:', checkpoint['epoch'], (score / total).item(), " ", vote_acc)

    @classmethod
    def baselineTest(cls, epoch, fold):
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
        checkpoints_pth = os.listdir('E:/ViT/')
        acc, acc_hat = 0, 0
        for cp in checkpoints_pth:
            if cp.startswith(f'{fold}test_{11 - fold}val_'):
                score, total = 0, 0
                checkpoint = torch.load(os.path.join('E:/ViT/', cp))
                if checkpoint['epoch'] == epoch:
                    trans.load_state_dict(checkpoint['state_dict'])
                    label_val, feature_val, lms3d_val, seqs_val = Dataprocess.loadSingleFold(11-fold, True)
                    val_dataloader = Dataprocess.baseline(71, feature_val, lms3d_val, label_val, True)
                    pred_lst, label_lst, belong_lst = [], [], []
                    for input, target, attribution in val_dataloader:
                        label_lst.append(target)
                        with torch.no_grad():
                            pred = trans(input, config['T_masked'])
                        pred = cls.softmax(pred)
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
                    print('epoch:', checkpoint['epoch'], (score / total).item(), " ", vote_acc)

    @classmethod
    def test4PickUpEpoch(cls, fold):
        from model import NormalB16ViT
        from train import config
        from dataProcess import Dataprocess
        import torch
        import os
        from dataProcess import Utils
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levwuelname)s: %(message)s',
                            level=logging.INFO,
                            filename=config['LOG_pth'],
                            filemode='a')
        # trans = MultiEmoTransformer(lms3dpro = config['T_lms3d_dim'], rgbpro = config['T_rgb_dim'], input=config['T_input_dim'], nhead=config['T_head_num'], num_layers=config['T_block_num'], batch_first=config['T_bs_first'], output_dim=config['T_output_dim'])
        trans = NormalB16ViT(weights=None)
        trans.cuda()
        trans.eval()
        acc_hat = 0
        writer_acc_val = SummaryWriter(f'./tb/acc/test4pick/pretrain/B16_OF')
        pos_encoding = Utils.SinusoidalEncoding(270, config['T_proj_dim'])
        for cp in range(382,1001,1): # 1test_0.pkl 2test_0.pkl ...
            if type(fold) is list:
                score10,total10,vote_acc10 = 0,0,0
                for fo in fold:
                    checkpoint = torch.load(os.path.join(config['checkpoint_pth'], f'{1}test_{cp}.pkl'))
                    trans.load_state_dict(checkpoint['state_dict'])
                    label_val,feature_val,lms3d_val,_, _,  = Dataprocess.loadSingleFoldOF(fo, True)
                    val_dataloader = Dataprocess.VideoNaive(config['window_size'], feature_val, lms3d_val, label_val, False, None)  # variables in memory
                    pred_lst, label_lst, belong_lst = [], [], []
                    for input, target  in val_dataloader: #attribution
                        label_lst.append(target)
                        # with torch.no_grad():
                        #     pred = trans(input)
                        with torch.no_grad():
                            pred = trans(input,pos_encoding)
                        pred = cls.softmax(pred)
                        pred_lst.append(pred)
                        # belong_lst.append(attribution)
                        idx_pred = torch.topk(pred, 1, dim=1)[1]
                        rs = idx_pred.eq(target.reshape(-1, 1))
                        score10 += rs.view(-1).float().sum()
                        total10 += input.shape[0]
                        print('current acc',score10/total10)
                logging.info(
                    f'vote highest:{acc_hat * 100}% when acc is {score10 / total10 * 100}%, epoch is {checkpoint["epoch"]}')
                writer_acc_val.add_scalar('test avg acc', score10 /total10 * 100, checkpoint['epoch'])
                # vote_acc10 += dataProcess.Utils.vote(pred_lst, belong_lst, label_lst)
                # writer_acc_val.add_scalar('test avg acc', vote_acc10/10*100, checkpoint['epoch'])
                if acc_hat < vote_acc10/10:
                    acc_hat = vote_acc10/10

                print('epoch:', cp, (score10 / total10).item()*100, " ", vote_acc10/10*100)
            else:
                if cp.startswith(f'{fold}test_'):
                    checkpoint = torch.load(os.path.join(config['checkpoint_pth'],cp))
                    trans.load_state_dict(checkpoint['state_dict'])
                    score5, total5 = 0, 0
                    label_val, feature_val, lms3d_val, seqs_val = Dataprocess.loadSingleFold(fold, True)
                    val_dataloader = Dataprocess.ConvertVideo2Samples100Votes(config['window_size'], feature_val, lms3d_val, label_val,True) # variables in memory
                    pred_lst,label_lst,belong_lst=[], [], []
                    for input, target, attribution in val_dataloader:
                        label_lst.append(target)
                        with torch.no_grad():
                            # pred = trans(input[:,:,512:],input[:,:,:512], config['T_masked'])
                            pred = trans(input)
                        pred = cls.softmax(pred)
                        pred_lst.append(pred)
                        belong_lst.append(attribution)
                        idx_pred = torch.topk(pred, 1, dim=1)[1]
                        rs = idx_pred.eq(target.reshape(-1, 1))
                        score5 += rs.view(-1).float().sum()
                        total5 += input.shape[0]
                    vote_acc5 = dataProcess.Utils.vote(pred_lst,belong_lst,label_lst)
                    score6, total6 = 0, 0
                    label_val, feature_val, lms3d_val, seqs_val = Dataprocess.loadSingleFold(11-fold, True)
                    val_dataloader = Dataprocess.ConvertVideo2Samples(config['window_size'], feature_val, lms3d_val, label_val,True) # variables in memory
                    pred_lst,label_lst,belong_lst=[], [], []
                    for input, target, attribution in val_dataloader:
                        label_lst.append(target)
                        with torch.no_grad():
                            # pred = trans(input[:,:,512:],input[:,:,:512], config['T_masked'])
                            pred = trans(input)
                        pred = cls.softmax(pred)
                        pred_lst.append(pred)
                        belong_lst.append(attribution)
                        idx_pred = torch.topk(pred, 1, dim=1)[1]
                        rs = idx_pred.eq(target.reshape(-1, 1))
                        score6 += rs.view(-1).float().sum()
                        total6 += input.shape[0]
                    vote_acc6 = dataProcess.Utils.vote(pred_lst,belong_lst,label_lst)
                    writer_acc_val.add_scalar('val acc', (vote_acc6+vote_acc5)/2 * 100, checkpoint['epoch'])
                    if acc_hat<(vote_acc6+vote_acc5)/2:
                        acc_hat = (vote_acc6+vote_acc5)/2
                        logging.info(f'vote highest:{acc_hat*100}% when acc is {(score5 / total5+score6/total6)/2*100}%, epoch is {checkpoint["epoch"]}')
                    print('epoch:',checkpoint['epoch'],(score5 / total5+score6/total6).item()/2," ",(vote_acc6+vote_acc5)/2)

    @classmethod
    def testlstmAB(cls, fold):
        from model import FollowLSTM
        from train import config
        from dataProcess import Dataprocess
        import torch
        import os
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levwuelname)s: %(message)s',
                            level=logging.INFO,
                            filename=config['LOG_pth'],
                            filemode='a')
        trans = FollowLSTM(inputDim=config['LSTM_input_dim'], hiddenNum=config['LSTM_hidden_dim'],
                           outputDim=config['LSTM_output_dim'], layerNum=config['LSTM_layerNum'],
                           cell=config['LSTM_cell'], use_cuda=config['use_cuda'])
        trans.cuda()
        trans.eval()
        writer_acc_val = SummaryWriter(f'./tb/acc/test4pick/pretrain/lstm_OF')
        writer_acc_vote = SummaryWriter(f'./tb/acc/test4pick/vote/lstm_OF')
        for cp in range(1, 20):  # 1test_0.pkl 2test_0.pkl ...
            score10, total10, vote_acc10 = 0, 0, 0
            pred_lst, label_lst, belong_lst = [], [], []
            for fo in fold:
                after = True
                if not os.path.exists(os.path.join(config['checkpoint_pth'], f'{fo}test_{cp}.pkl')):
                    after = False
                    continue
                checkpoint = torch.load(os.path.join(config['checkpoint_pth'], f'{fo}test_{cp}.pkl'))
                trans.load_state_dict(checkpoint['state_dict'])
                val_dataloader = Dataprocess.loadFERModelIntoDataloader(fo, 'test')
                for input, target, attribution in tqdm.tqdm(val_dataloader):  # attribution
                    with torch.no_grad():
                        pred = trans(input)
                    pred = cls.softmax(pred)
                    i_mask = pred.max(dim=1)[0] > torch.tensor([0.7] * input.shape[0]).cuda()
                    pred_lst.append(pred[i_mask])
                    label_lst.append(target[i_mask])
                    belong_lst.append(attribution[i_mask])
                    idx_pred = torch.topk(pred, 1, dim=1)[1]
                    rs = idx_pred.eq(target.reshape(-1, 1))
                    score10 += rs.view(-1).float().sum()
                    total10 += input.shape[0]
                    print('current acc', score10 / total10)
            if after:
                vote_acc10 += dataProcess.Utils.vote(pred_lst, belong_lst, label_lst)
                writer_acc_val.add_scalar('test avg acc', score10 / total10 * 100, cp)
                writer_acc_vote.add_scalar('test vote acc', vote_acc10 * 100, cp)
                print('acc ', score10 / total10, 'vote', vote_acc10)
    @classmethod
    def testCheck(cls,fold):
        from model import NormalB16ViT
        from train import config
        from dataProcess import Dataprocess
        import torch
        import os
        from dataProcess import Utils
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levwuelname)s: %(message)s',
                            level=logging.INFO,
                            filename=config['LOG_pth'],
                            filemode='a')
        trans = NormalB16ViT(weights=None)
        trans.cuda()
        trans.eval()
        writer_acc_val = SummaryWriter(f'./tb/acc/test4pick/pretrain/B16_OF')
        writer_acc_vote = SummaryWriter(f'./tb/acc/test4pick/vote/B16_OF')
        pos_encoding = Utils.SinusoidalEncoding(50, config['T_proj_dim'])
        for cp in range(3, 19):  # 1test_0.pkl 2test_0.pkl ...
            for iter in range(10,2500,10):
                score10, total10, vote_acc10 = 0, 0, 0
                pred_lst, label_lst, belong_lst = [], [], []
                for fo in fold:
                    after = True
                    if not os.path.exists(os.path.join(config['checkpoint_pth'], f'{fo}test_{cp}_{iter}.pkl')):
                        after = False
                        continue
                    checkpoint = torch.load(os.path.join(config['checkpoint_pth'], f'{fo}test_{cp}_{iter}.pkl'))
                    trans.load_state_dict(checkpoint['state_dict'])
                    val_dataloader = Dataprocess.loadFERModelIntoDataloader(fo, 'test')
                    for input, target, attribution in tqdm.tqdm(val_dataloader):  # attribution
                        label_lst.append(target)
                        with torch.no_grad():
                            pred = trans(input, pos_encoding)
                        pred = cls.softmax(pred)
                        pred_lst.append(pred)
                        belong_lst.append(attribution)
                        idx_pred = torch.topk(pred, 1, dim=1)[1]
                        rs = idx_pred.eq(target.reshape(-1, 1))
                        score10 += rs.view(-1).float().sum()
                        total10 += input.shape[0]
                        print('current acc', score10 / total10)
                if after:
                    vote_acc10 += dataProcess.Utils.vote(pred_lst, belong_lst, label_lst)
                    writer_acc_val.add_scalar('test avg acc', score10 / total10 * 100, iter)
                    writer_acc_vote.add_scalar('test vote acc', vote_acc10 * 100, iter)
                    print('acc ', score10/total10, 'vote', vote_acc10)
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
    # Modeltest.test(877,5)
    # deleteStaticDict('./',0)
    # Modeltest.baselineVal(5)
    # Modeltest.baselineTest(1000, 5)
    # folds = [i+1 for i in range(1)]
    # Modeltest.test4PickUpEpoch(fold=folds)
    # Modeltest.lstmtest([1])
    Modeltest.testlstmAB([1,2,3])