import torch
from dataProcess import Dataprocess
from container import Models
import yaml

config = yaml.safe_load(open('./config.yaml'))


def train():
    m = Models(train=True)
    optimizer = torch.optim.SGD(m.LSTM_model.parameters(), lr=float(config['learning_rate']), momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    for fold in range(1, 11):
        label, feature, lms3d, seqs, _, _, _, _ = Dataprocess.dataForLSTM(fold)
        start_idx = 0
        input = torch.zeros((0,int(config['LSTM_input_dim']))).cuda()
        target = torch.empty(0).cuda()
        for e in range(30):
            for sqs in range(seqs.shape[0]):
                sequense_label = label[start_idx:start_idx + int(seqs[sqs])]
                sequense_feature = feature[start_idx:start_idx + int(seqs[sqs])]
                sequense_lms3d = lms3d[start_idx * 68:(start_idx + int(seqs[sqs])) * 68].reshape(int(seqs[sqs]), -1)
                start_idx = int(seqs[sqs])
                if seqs[sqs] < 9:
                    continue
                else:  # 一个帧长合格的视频，window size = 9 , 切割视频
                    for idx in range(sequense_feature.shape[0] - int(config['window_size']) + 1):
                        sample_label = sequense_label[idx:int(idx + config['window_size'])]
                        sample_feature = sequense_feature[idx:int(idx + config['window_size'])]
                        feature_mean = torch.mean(sample_feature,dim=1)
                        feature_std = torch.std(sample_feature,dim=1)
                        sample_feature = (sample_feature - feature_mean.reshape(-1,1))/feature_std.reshape(-1,1)
                        sample_lms3d = sequense_lms3d[idx:int(idx + config['window_size'])]
                        lms_mean = torch.mean(sample_lms3d,dim=1)
                        lms_std = torch.std(sample_lms3d,dim=1)
                        sample_lms3d = (sample_lms3d - lms_mean.reshape(-1, 1)) / lms_std.reshape(-1, 1)
                        sample_input = torch.cat((sample_feature, sample_lms3d), dim=1)
                        input = torch.cat((input, sample_input), dim=0)
                        # input = torch.cat((input, sample_feature), dim=0)
                        target = torch.cat((target, sample_label), dim=0)
                        rs_input = input.reshape(-1,int(config['window_size']),int(config['LSTM_input_dim']))
                        rs_target = target.reshape(-1,int(config['window_size']))
                        if rs_input.shape[0]==int(config['batch_size']):
                            optimizer.zero_grad()
                            output = m.LSTM_model(rs_input)
                            # print(rs_input)
                            # loss = loss_func(torch.topk(output,1,dim=1)[1][:,0].to(torch.float), rs_target[:,0].to(torch.float))
                            loss = loss_func(output,rs_target[:,0].long())
                            loss.requires_grad_(True)
                            loss.backward()
                            optimizer.step()
                            input = torch.zeros((0, int(config['LSTM_input_dim']))).cuda()
                            target = torch.empty(0).cuda()
                            print(loss)
                            print('==============')

if __name__ == '__main__':
    train()
