import os
import random

import cv2
import numpy as np
import torch
import tqdm

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
    s = torch.nn.Softmax(dim=1)
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
def testtensor():
    a = [torch.tensor([[1],[2],[3],[4]]),torch.tensor([[1],[2],[3],[4]])]
    print(sum(a))
    b = torch.zeros((8,7))
    c = torch.ones((8,7))
    d = [b,c]
    d = torch.cat(d)
    print(d.shape)
def testtopk():
    a = torch.tensor((1,1,1))
    b = torch.topk(a,2,dim=0)[1]
    print(a, b)
def voteunitest():
    from dataProcess import Utils
    a = [torch.rand((3,7)),torch.rand((1,7)),torch.rand((2,7))]
    print(a[0])
    print(a[1])
    print(a[2])
    b = [torch.tensor((0,)),torch.tensor((0,)),torch.tensor((0,)),torch.tensor((1,)),torch.tensor((2,)),torch.tensor((2,))]
    c = [torch.tensor((0,)),torch.tensor((1,)),torch.tensor((2,))]
    Utils.vote(a,b,c)
def testtesnor():
    a = torch.tensor([True,False,True,False,True])
    print(a.device)
def testtensor():
    a = torch.tensor(123,device='cuda')
    b = a.item()
    print(b.device)
def testlistrm():
    f_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for a in range(1,11):
        ori = f_lst.copy()
        ori.remove(a)
        ori.remove(11-a)
        print(ori,a,11-a)
def evenly():
    video = np.arange(42)
    video_index = np.zeros((video.shape[0],))
    video_shuffle = video.copy()
    np.random.shuffle(video_shuffle)
    print(video_shuffle)
    np.random.shuffle(video_shuffle)
    print(video_shuffle)
    sample_index = video_shuffle[:10]
    print(sample_index)
    video_index[sample_index] = 1
    print(video[video_index.astype(int).astype(bool)])
    one_sample = video[video_index.astype(int).astype(bool)]
def combination():
    import math
    print(math.factorial(20)//(math.factorial(10)*math.factorial(10)))
    print(math.factorial(4))
def axismax():
    a = np.arange(16).reshape((4,4))
    print(a.max(axis=1))
def drawHist():
    import matplotlib.pyplot as plt
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    me_f = np.loadtxt('./dataset/fan_feature_fold1_test.txt')
    q_f = np.loadtxt('./dataset/feature1.txt')
    x_axis_data = [i for i in range(10)]
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis_data, me_f[0][:10], 'ro-', color='r', alpha=0.8, linewidth=1, label='一些数字')
    plt.plot(x_axis_data, q_f[0][:10], 'ro-', color='g', alpha=0.8, linewidth=1, label='一些数字')
    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('x轴数字')
    plt.ylabel('y轴数字')
    plt.show()
    return
def countsamplemaxnumber():
    mamm = 0
    for i in range(1,11):
        a = np.loadtxt(f'./dataset/split_{i}_test.txt')
        b = np.loadtxt(f'./dataset/split_{i}_train.txt')
        print(a.max(),b.max())
def sample():
    sam_idx = sorted(random.sample([S for S in range(71)], 71))
    print(sam_idx == [S for S in range(71)])
def slice():
    a = np.array([1,23,4])
    print(a[:6])
def observeData():
    lms = np.loadtxt('./dataset/3dlms_fold1_test.txt')
    rgb = np.loadtxt('./dataset/feature1.txt')
    lms0 = lms[:68].reshape((1,204))
    rgb0 = rgb[0].reshape((1,512))
    meanl = np.mean(lms0,axis=1)
    # meanr = np.mean(rgb0,axis=1)
    stdl = np.std(lms0,axis=1)
    # stdr = np.std(rgb0,axis=1)
    lms0 = (lms0 - meanl)/stdl
    # rgb0 = (rgb0 - meanr)/stdr
    print(lms0)
    print(rgb0)

def get_current_lr(optimizer, group_idx, parameter_idx):
    # Adam has different learning rates for each paramter. So we need to pick the
    # group and paramter first.
    group = optimizer.param_groups[group_idx]
    p = group['params'][parameter_idx]

    beta1, _ = group['betas']
    state = optimizer.state[p]

    bias_correction1 = 1 - beta1 ** state['step']
    current_lr = group['lr'] / bias_correction1
    return current_lr

def getlr():
    import torch
    import timm.optim.optim_factory as optim_factory
    from model import MAEEncoder
    trans = MAEEncoder(embed_dim=512, depth=6, num_heads=8)
    trans.train()
    trans.cuda()
    param_groups = optim_factory.add_weight_decay(trans, 0.05)
    optimizer = torch.optim.Adam(param_groups, lr=1e-3, betas=(0.9, 0.95))
    mockinput = torch.rand((64,10,340)).cuda()
    loss_func = torch.nn.CrossEntropyLoss()
    target = torch.zeros((64,)).cuda()
    for e in range(2000):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        for itt in range(460):
            optimizer.zero_grad()
            pred = trans(mockinput)
            loss = loss_func(pred, target.long())
            loss.backward()
            optimizer.step()
            print(get_current_lr(optimizer,0,0))
def deletepth():
    import tqdm
    for i in tqdm.tqdm(range(1,11)):
        for e in tqdm.tqdm(range(2000,4000)):
            os.remove(f'/home/exp-10086/Project/Data/ViT/{i}test_{e}.pkl')
def testnpeinsum():
    a = np.einsum('i, j->ij',np.zeros((2,)),np.zeros((3,)))
    print(a)
def testnumpyinsert():
    a = np.arange(16).reshape(4,4)
    b = np.eye(4)
    c = list(zip(a,b))
    print(np.array(c).reshape(-1,4))
def OFmaxMin():
    for i in range(1,6):
        print(np.max(np.loadtxt(f'/home/exp-10086/Project/ferdataset/ourFace/seq/seq_{i}.txt')))
        print(np.min(np.loadtxt(f'/home/exp-10086/Project/ferdataset/ourFace/seq/seq_{i}.txt')))
        print(np.median(np.loadtxt(f'/home/exp-10086/Project/ferdataset/ourFace/seq/seq_{i}.txt')))
def paramintotxt(fold):
    fold_root = '/home/exp-10086/Project/Emotion-FAN-master/data/txt/CK+_10-fold_sample_IDascendorder_step10.txt'
    fold_fp = open(fold_root)
    lines = fold_fp.readlines()
    start = 0
    for lidx, l in enumerate(lines):
        if l.__contains__(f'{fold}-fold'):
            vn = l.split(' ')[1]
            start = lidx
            break
        else:
            continue
    fold_lines = lines[start + 1:start + int(vn) + 1]
    root = '/home/exp-10086/Project/ferdataset/ck_alpha_param'

    for params in fold_lines:
        ps = os.listdir(os.path.join(root, params.split(' ')[0]))
        ps.sort()
        for p in ps:
            pth = os.path.join(root, params.split(' ')[0], p)
            param = np.loadtxt(pth)
            shape = param[:40]
            exp = param[40:]
            shapes.append(shape)
            exps.append(exp)
def geryparamtxt(fold):
    fold_root = '/home/exp-10086/Project/Emotion-FAN-master/data/txt/CK+_10-fold_sample_IDascendorder_step10.txt'
    fold_fp = open(fold_root)
    lines = fold_fp.readlines()
    start = 0
    for lidx, l in enumerate(lines):
        if l.__contains__(f'{fold}-fold'):
            vn = l.split(' ')[1]
            start = lidx
            break
        else:
            continue
    fold_lines = lines[start + 1:start + int(vn) + 1]
    root = '/home/exp-10086/Project/ferdataset/ck_alpha_param'
    with open(f'/home/exp-10086/Project/ferdataset/ck_alpha_param/param_{fold}.txt','a') as pf:
        for params in fold_lines:
            ps = os.listdir(os.path.join(root, params.split(' ')[0]))
            ps.sort()
            for p in ps:
                pth = os.path.join(root, params.split(' ')[0], p)
                param = np.loadtxt(pth)
                shape = param[:40]
                exp = param[40:]
                shape = (shape - (-10307.826988220313)) / 78179.57736484919
                exp = (exp - (-0.13336482156029097)) / 0.7782772432072191
                np.savetxt(pf, np.hstack((shape,exp)))
def checkseq():
    path = '/home/exp-10086/Project/ferdataset/ourFace/seq/seq_1.txt'
    a = np.loadtxt(path)
    print(np.sum(a))
def countAFEWseq():# 0-151
    train_root = '/home/exp-10086/Project/Emotion-FAN-master/data/face/train_afew/'
    emos = os.listdir(train_root)
    seq_len = []
    for e in emos:
        vids = os.listdir(os.path.join(train_root, e))
        for v in vids:
            frames = os.listdir(os.path.join(train_root, e, v))
            seq_len.append(len(frames))
    print(min(seq_len),max(seq_len))
from sklearn import svm,metrics,preprocessing
import torchvision.transforms as transforms
from PIL import Image
import pickle
INPUT_SIZE = (224, 224)
ALL_DATA_DIR = '/home/exp-10086/Project/Emotion-FAN-master/data/video/'
PATH='/home/exp-10086/Project/face-emotion-recognition-main/models/affectnet_emotions/enet_b0_8_best_afew.pt'
MODEL2EMOTIW_FEATURES='./enet0_8_afew_pt_feat_emotiw.pickle'
model = torch.load(PATH)
feature_extractor_model = torch.load(PATH)
feature_extractor_model.classifier=torch.nn.Identity()
feature_extractor_model.eval()
feature_extractor_model.cuda()
DATA_DIR=os.path.join(ALL_DATA_DIR)
emotion_to_index = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Neutral':4, 'Sad':5, 'Surprise':6}
device = 'cuda'
test_transforms = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)
def get_features(data_dir):
    filename2features = {}
    for filename in tqdm.tqdm(os.listdir(data_dir)):
        frames_dir = os.path.join(data_dir, filename)
        X_global_features, X_isface = [], []
        imgs = []
        for img_name in os.listdir(frames_dir):
            # img = Image.open(os.path.join(frames_dir, img_name))
            img = Image.open('/home/exp-10086/Project/ferdataset/ourFace/vote_frame_cropped/Anger/10/1.jpg')
            img_tensor = test_transforms(img)
            X_isface.append('noface' not in img_name)
            if img.size:
                imgs.append(img_tensor)
                if len(imgs) >= 16:
                    scores = feature_extractor_model(torch.stack(imgs, dim=0).to(device))
                    scores = scores.data.cpu().numpy()
                    if len(X_global_features) == 0:
                        X_global_features = scores
                    else:
                        X_global_features = np.concatenate((X_global_features, scores), axis=0)
                    imgs = []
        if len(imgs) > 0:
            scores = feature_extractor_model(torch.stack(imgs, dim=0).to(device))
            scores = scores.data.cpu().numpy()
            if len(X_global_features) == 0:
                X_global_features = scores
            else:
                X_global_features = np.concatenate((X_global_features, scores), axis=0)
        X_isface = np.array(X_isface)
        filename2features[filename] = (X_global_features, X_isface)
    return filename2features
def generateFea():
    filename2features_val = get_features(os.path.join(DATA_DIR, 'val_afew/AlignedFaces_LBPTOP_Points/frames_mtcnn_cropped/'))
    filename2features_train = get_features(os.path.join(DATA_DIR, 'train_afew/AlignedFaces_LBPTOP_Points/frames_mtcnn_cropped/'))  # _cropped

    with open(MODEL2EMOTIW_FEATURES, 'wb') as handle:
        pickle.dump([filename2features_train,filename2features_val], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(MODEL2EMOTIW_FEATURES)
###############################################################################################################################
def create_dataset(filename2features, data_dir):
    x = []
    y = []
    has_faces = []
    for category in emotion_to_index:
        for filename in os.listdir(os.path.join(data_dir, category)):
            fn = os.path.splitext(filename)[0]
            if not fn in filename2features:
                continue
            features = filename2features[fn]
            total_features = None
            if True:
                cur_features = features[0][features[-1] == 1]
            else:
                cur_features = features[0]
            if len(cur_features) == 0:
                has_faces.append(0)
                total_features = np.zeros_like(feature)
            else:
                has_faces.append(1)
                mean_features = (np.mean(cur_features, axis=0))
                std_features = (np.std(cur_features, axis=0))
                max_features = (np.max(cur_features, axis=0))
                min_features = (np.min(cur_features, axis=0))
                feature = np.concatenate((mean_features, std_features, min_features, max_features), axis=None)
                total_features = feature
            if total_features is not None:
                x.append(total_features)
                y.append(emotion_to_index[category])
    x = np.array(x)
    y = np.array(y)
    has_faces = np.array(has_faces)
    return x, y, has_faces
def afewTrainandTest():
    with open(MODEL2EMOTIW_FEATURES, 'rb') as handle:
        filename2features_train,filename2features_val=pickle.load(handle)
    print(len(filename2features_train),len(filename2features_val))
    x_test, y_test, has_faces_test = create_dataset(filename2features_val, os.path.join(DATA_DIR, 'val_afew'))
    x_train, y_train, has_faces_train = create_dataset(filename2features_train, os.path.join(DATA_DIR, 'train_afew'))
    x_train_norm=preprocessing.normalize(x_train,norm='l2')
    x_test_norm=preprocessing.normalize(x_test,norm='l2')
    clf = svm.LinearSVC(C=1.1)
    clf.fit(x_train_norm[has_faces_train==1], y_train[has_faces_train==1])
    y_pred = clf.predict(x_test_norm)
    print("Accuracy:",metrics.accuracy_score(y_test[has_faces_test==1], y_pred[has_faces_test==1]))
    print("Complete accuracy:",metrics.accuracy_score(y_test, y_pred))

def countlmsandrgbAndfixLms():
    rgb_ = '/home/exp-10086/Project/ferdataset/ourFace/vote_frame_cropped/'
    lms_ = '/home/exp-10086/Project/ferdataset/ourFace/votelabel_lms/'
    for e in os.listdir(rgb_):
        vis = os.listdir(f'{rgb_}{e}/')
        for v in vis:
            frames = os.listdir(f'{rgb_}{e}/{v}')
            for f in frames:
                if not os.path.exists(f'{lms_}{e}/{v}/{f[:-4].replace("noface","")}.txt'):
                    lost_lms_f = f'{lms_}{e}/{v}/{f[:-4].replace("noface","")}.txt'
                    pre_lms,post_lms = None,None
                    for i in range(1,271):
                        pre_lms_f = f'{lms_}{e}/{v}/{int(f[:-4].replace("noface",""))-i}.txt'
                        if os.path.exists(pre_lms_f):
                            pre_lms = np.loadtxt(pre_lms_f)
                            break
                    for j in range(1,271):
                        post_lms_f = f'{lms_}{e}/{v}/{int(f[:-4].replace("noface",""))+j}.txt'
                        if os.path.exists(post_lms_f):
                            post_lms = np.loadtxt(post_lms_f)
                            break
                    if pre_lms is not None and post_lms is not None:
                        dis = j+i
                        lost_lms = i/dis * pre_lms + j/dis * post_lms
                    if pre_lms is None and post_lms is not None:
                        lost_lms = post_lms
                    if pre_lms is not None and post_lms is None:
                        lost_lms = pre_lms
                    if pre_lms is not None and post_lms is not None:
                        pass
                    # np.savetxt(lost_lms_f,lost_lms)
                    print(pre_lms_f,lost_lms_f,post_lms_f)
if __name__ == '__main__':
    # shapes, exps = [], []
    # for fold in range(1,11):
    #     geryparamtxt(fold)
    # print(np.std(np.array(shapes)), np.mean(np.array(shapes)), np.std(np.array(exps)), np.mean(np.array(exps)))
    countlmsandrgbAndfixLms()
    pass
