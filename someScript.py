import numpy as np
import torch


def checkCKplusDataset():
    import os
    count = 0
    ca = os.listdir('E:/cohn-kanade-images/')
    for s in ca:
        sub_ca1 = os.path.join('E:/cohn-kanade-images/', s)
        sub_ca = os.listdir(sub_ca1)
        for seq in sub_ca:
            d = os.path.join(sub_ca1, seq)
            if os.path.isdir(d):
                count += 1
    print(count)
    '''总数量正确，带label数据集已经由emotion-FAN确认'''


def dicYaml():
    # yaml中存储dict数据类型测试
    import yaml
    cfg = yaml.safe_load(open('./config.yaml', 'r', encoding='utf-8').read())
    print(cfg['CK+_dict']['Happy'])
    print(type(float(cfg['learning_rate'])))

def countSequenceLen():
    import os
    min_len = []
    wenjianjia = os.listdir('E:/cohn-kanade-images/')
    labeledf = open('./dataset/CK+_10fold_samples.txt','r',encoding='utf-8')
    labeled = labeledf.readlines()
    datal = []
    for l in labeled:
        datal.append(l.split(' ')[0])
    print(datal)
    for w in wenjianjia:
        bianhao = os.listdir(os.path.join('E:/cohn-kanade-images/', w))
        for b in bianhao:
            conStr = f'{w}/{b}'
            frame = os.listdir(os.path.join('E:/cohn-kanade-images/', w,b))
            if conStr in datal:
                min_len.append(len(frame))
            if len(frame)==29:
                pass
                # print(bianhao,frame)
    print(sorted(min_len))
    return sorted(min_len)
def testModelId():
    from container import Models
    m = Models(False)
    print(id(m.LSTM_model))
    print(id(m.LSTM_model))
def testEQFunc():
    a = torch.zeros((16, 1), dtype=torch.int)
    b = torch.zeros((16, 1), dtype=torch.float32)
    c = b.eq(a)
    print(c.float().sum())
def testBCE():
    import torch.nn as nn
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss(m(input), target)
    output.backward()
    print(input)
    print(target)
def plotPics():
    from matplotlib import pyplot as plt
    statistits = countSequenceLen()
    plt.hist(statistits, 100)
    plt.xlabel('frame_number')
    plt.ylabel('video_number')
    plt.show()
def statistitsFrames():
    plotPics()
    statistits = countSequenceLen()
    for i in range(0, 80, 10):
        fs = [s for s in statistits if i < s <= 10 + i]
        print(f'{i}到{i + 10}帧数的视频数量为{len(fs)}')
    print(f'视频总数为{len(statistits)}')
def testtile():
    a = np.zeros((7,512))
    b = np.tile(a,(2,1))
    print(b.shape)
def drawNormal():
    rs = np.random.normal(0,1,15000)
    print(type(rs))
    print(rs)
    from matplotlib import pyplot as plt
    plt.hist(rs,15000)
    plt.show()

if __name__ == '__main__':
    testtile()
    pass
