import random

import cv2
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
def testModelParams():
    checkpoint = torch.load('./weights/self_relation-attention_fold_12_90.9091')
    FAN_state_dict = checkpoint['state_dict']
    print(FAN_state_dict.keys())
def testAverage():
    a = [0] * 10
    for j in range(10):
        for i in range(99):
            a[j] = i
    print(sum(a) / len(a))
def testImageValue():
    from PIL import Image
    import torchvision.transforms as transforms
    import cv2
    img = Image.open('E:/cohn-kanade-images/S130/007/S130_007_00000001.png').convert('RGB')
    mat = transforms.Compose([transforms.ToTensor()])
    mimg = mat(img)
    print(mimg[0][0][0])
    img0 = cv2.imread('E:/cohn-kanade-images/S130/007/S130_007_00000001.png')
    img1 = np.swapaxes(img0, 0, 2)
    img2 = np.swapaxes(img1, 1, 2)
    print(img2[0][0][0] / 255)
def testprint():
    print('a{b:.4f} ({b:.4f})'.format(b=2,c=3))

def testsqueeze():
    a=np.zeros((1,3))
    b=np.zeros((2,3))
    c=np.zeros((3,3))
    d = np.array([a,b,c], dtype=object)
    e = np.concatenate(d,axis=0)
    print(e.shape)
def teststep():
    a = [i for i in range(25)]
    b = [0]*25
    c = []
    e = [a]*2
    for l in range(len(e)):
        for st in range(3):
            d = []
            for i in range(st, 25, 3):
                d.append(e[l][i])
            c.append(d)
            print(len(d))
    print(c)
def testlist():
    a = [1,2,3]
    a.append(a[-1])
    print(a)
def testmul():
    import torch.nn as nn
    a = torch.ones((2,25,204))
    c = nn.Linear(204,1)
    print(c(a).shape)
def testadd():
    a = torch.ones((8,25,1))
    b = torch.ones((8,25,460))
    print(torch.mul(a, b).shape)
def testencoder():
    import torch.nn as nn
    encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
    src = torch.rand(32, 10, 256)
    out = encoder_layer(src)
    print(out.shape)
def testclstoken():
    import torch.nn as nn
    cls = nn.Parameter(torch.randn(1, 1, 5))
    x = torch.ones((8,25,204))
    print(torch.tile(cls,(8,1,1)).shape)
def testbroadcast():
    from torch.autograd import Variable
    a = torch.ones((1,26,204))
    b = torch.zeros((8,26,204))
    c = b + Variable(a[:,:b.size(1)],requires_grad=False)
    print(c)
def testtensorboard():
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter('./loss/')
    for i in range(100):
        a = torch.tensor(1,dtype=torch.float32)
        w.add_scalar('loss',a,i)
def testsincos():
    pe = torch.arange(0, 26).unsqueeze(1)
    print(pe[:, 0::2])
    print(pe[:, 1::2])
def testsoftmax():
    a = torch.arange(20,dtype=torch.float32).reshape((4,5))
    c = torch.arange(20,dtype=torch.float32).reshape((4,5))
    b = torch.arange(20,dtype=torch.float32).reshape((5,4))
    d = torch.mm(a,b)
    s = torch.nn.Softmax(dim=0)
    e = s(d)
    print(e)
    print(torch.mm(e,c))
def testtype():
    a = []
    print(type(a)==list)
def testnprandom():
    a = random.sample([i for i in range(17)], 12)
    print(type(a))
def testdim():
    a = np.zeros((2,3,4))
    b = np.ones((3,4))
    a[0] = b
    print(a)
def testdataset():
    from dataProcess import LSTMDataSet
    a = torch.zeros((3,4,5))
    b = torch.zeros((3,4,5))
    dataset =  LSTMDataSet(a,b)
    train = dataset[:2]
    test = dataset[2]
    print(dataset.input.shape[0])
def solveICPBySVD(p0, p1):
  """svd求解3d-3d的Rt, $p_0 = Rp_1 + t$
  Args:
    p0 (np.ndarray): nx3
    p1 (np.ndarray): nx3
  Returns:
    R (np.ndarray): 3x3
    t (np.ndarray): 3x1
  """
  p0c = p0.mean(axis=0, keepdims=True) # 质心
  p1c = p1.mean(axis=0, keepdims=True) # 1, 3
  q0 = p0 - p0c
  q1 = p1 - p1c # n x 3
  W = q0.T @ q1 # 3x3
  U, _, VT = np.linalg.svd(W)
  R = U @ VT # $UV^T$
  t = p0c.T - R @ p1c.T
  return R, t
def testSVDIcp():
    from dataProcess import Utils,config
    from container import Models
    t = Models()
    sPC = Utils.standardPC(t) # fixed face pose
    random_idx = random.sample([i for i in range(config['PC_points_sample_range'])], config['PC_points_piars'])
    standard_idx = sPC[random_idx] # random point in standard face
    standard_distance = np.linalg.norm(standard_idx[:int(config['PC_points_piars'] / 2)] - standard_idx[int(config['PC_points_piars'] / 2):], axis=1)
    _, _, ver_lst, _, pc = Utils.deep_features('E:/cohn-kanade-images/S005/001/S005_001_00000001.png', 0, t)
    _, _, ver_lst0, _, pc0 = Utils.deep_features('E:/cohn-kanade-images/S149/002/S149_002_00000013.png', 0, t)
    vari_idx = pc[random_idx]
    vari_idx0 = pc0[random_idx]
    vari_distance = np.linalg.norm(vari_idx[:int(config['PC_points_piars'] / 2)] - vari_idx[int(config['PC_points_piars'] / 2):], axis=1)
    vari_distance0 = np.linalg.norm(vari_idx0[:int(config['PC_points_piars'] / 2)] - vari_idx0[int(config['PC_points_piars'] / 2):], axis=1)
    rates = standard_distance / vari_distance
    rates0 = standard_distance / vari_distance0
    rate = np.median(rates)
    rate0 = np.median(rates0)
    ver_lst[0] *= rate
    ver_lst0[0] *= rate0
    pc *= rate
    pc0 *= rate0
    Utils.open3dVerify(np.hstack((ver_lst[0], ver_lst0[0])), np.vstack((pc, pc0)), sPC)
    r,t = solveICPBySVD(p0=sPC,p1 = pc)
    r0,t0 = solveICPBySVD(p0=sPC,p1 = pc0)
    newpc = r@pc.T+t
    newpc0 = r0@pc0.T+t0
    newlms = r @ ver_lst[0] + t
    newlms0 = r0 @ ver_lst0[0] + t0
    Utils.open3dVerify(np.hstack((newlms,newlms0)),np.vstack((newpc.T,newpc0.T)),sPC)
def testnprandomint():
    raw_lms_frame = np.zeros((68,3))
    how_many_point = np.random.randint(1, high=69)
    lms_idx = random.sample([i for i in range(68)], how_many_point)
    for l in lms_idx:
        value_times = random.sample([10,100,1000],1)
        noi_xyz = np.random.random((1,3)) * value_times
        raw_lms_frame[l] = noi_xyz
        print(raw_lms_frame)
def testCVDoG():
    src = cv2.imread('./img.png')
    tg0 = cv2.GaussianBlur(src,(5,5),0.3)
    tg1 = cv2.GaussianBlur(src,(5,5),0.4)
    dog = abs(tg1-tg0)
    cv2.imshow('0', tg0)
    cv2.imshow('1', tg1)
    cv2.imshow('dog',dog)
    cv2.waitKey(0)
def testlist2np():
    a = [np.zeros((2,3))]
    print(np.concatenate(a,axis=0))
def testembedding():
    emb = torch.nn.Embedding(2,64)
    print(emb)
def testtensorsplit():
    a = torch.zeros((8,50,716))
    c = torch.ones((8,716))
    a[:,24,:] = c
    print(a.shape)
    b= torch.split(a,int(a.shape[1]/2),dim=1)
    print(len(b))
def testcat():
    a = torch.zeros(8,25,204)
    b = torch.zeros(8,1,204)
    c = torch.cat((a,b),dim = 1)
    print(c.shape)
def testgt():
    a = torch.zeros((3,2,4))
    b = torch.ones((1,2,4))
    c = -torch.ones((1,2,4))
    a[0] = b
    a[1] = c
    d = a==0
    e = torch.sum(d,dim=2)
    print(e!=0)
def test0mulinf():
    a = 0.1
    b = -1e+10
    print(a*b)
def testRotateMatrix():
    from dataProcess import target
    eye = np.array([[10,22],[30,34]])
    RotateMatrix, _ = cv2.estimateAffinePartial2D(eye, target)
    print(RotateMatrix)
if __name__ == '__main__':
    testRotateMatrix()
    pass
