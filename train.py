import torch
from dataProcess import Dataprocess, LSTMDataSet
from model import LSTMModel
import yaml
from tqdm import trange
from dataProcess import EOS
config = yaml.safe_load(open('./config.yaml'))


def train_single_vedio():
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


def train_mini_batch():
    for fold in trange(1, 11):
        lstm = LSTMModel(inputDim=config['LSTM_input_dim'], hiddenNum=config['LSTM_hidden_dim'],
                         outputDim=config['LSTM_output_dim'], layerNum=config['LSTM_layerNum'],
                         cell=config['LSTM_cell'], use_cuda=config['use_cuda'])
        lstm.cuda()
        optimizer = torch.optim.RMSprop(lstm.parameters(), lr=float(config['learning_rate']), momentum=0.9)
        loss_func = torch.nn.CrossEntropyLoss()
        lstm.train()
        label, feature, lms3d, _, label_test, feature_test, lms3d_test, _ = Dataprocess.dataForLSTM(fold, crop=True)
        train_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'],feature,lms3d,label)
        test_dataloader = Dataprocess.dataAlign2WindowSize(config['window_size'],feature_test,lms3d_test,label_test)
        # input = torch.zeros((0, int(config['window_size']), int(config['LSTM_input_dim']))).cuda()
        # target = torch.empty(0).cuda()
        for e in trange(1024):
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
            # if score/total == 1:
                # torch.save({'state_dict': lstm.state_dict(),'EOS':EOS},
                #     f'./weights/minibatch_fold_{fold}_epoch_{e}_acc_{score / total}_bs_{config["batch_size"]}_lr_{config["learning_rate"]}_ln_{config["LSTM_layerNum"]}_hd_{config["LSTM_hidden_dim"]}_ws_{config["window_size"]}.pth')
                # break
        # eval
        score = 0
        total = 0
        lstm.eval()
        for test_input,test_label in test_dataloader:
            pred = lstm(test_input)
            idx_pred = torch.topk(pred, 1, dim=1)[1]
            rs = idx_pred.eq(test_label.reshape(-1, 1))
            score += rs.view(-1).float().sum()
            total += test_input.shape[0]
        print(f'eval acc:{score/total*100}%')
def main():
    import argparse
    parser = argparse.ArgumentParser(description='LSTM train function choice')
    parser.add_argument('-M', default='m', type=str, metavar='N',
                        help='s means single and m means minibatch')
    args = parser.parse_args()
    if args.M == 's':
        train_single_vedio()
    if args.M == 'm':
        train_mini_batch()


if __name__ == '__main__':
    main()
