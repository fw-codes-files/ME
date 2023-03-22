'''
Neural Network models, implemented by PyTorch
'''
import time

import numpy as np
import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import yaml

config = yaml.safe_load(open('./config.yaml', encoding='utf-8'))
# RNNs模型基类，主要是用于指定参数和cell类型
class BaseModel(nn.Module):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda=False):

        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        self.use_cuda = use_cuda
        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.hiddenNum,
                               num_layers=self.layerNum, dropout=0.0,
                               nonlinearity="tanh", batch_first=True, )
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                                num_layers=self.layerNum, dropout=0.0,
                                batch_first=True, )

        self.fc = nn.Linear(1024, self.outputDim)
# 标准RNN模型
class RNNModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda):
        super(RNNModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda)

    def forward(self, x):
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        if self.use_cuda:
            h0 = h0.cuda()
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput
# LSTM模型
class LSTMModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda, feature_dim=1):
        super(LSTMModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda)
        self.input2feature = nn.Sequential(nn.Linear(inputDim, 512), nn.ReLU(), nn.Linear(512, feature_dim), nn.ReLU())
        self.alpha = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())  # act on single frame
        self.beta = nn.Sequential(nn.Linear(feature_dim + hiddenNum, 1), nn.Sigmoid())  # act on single sample
        self.lastL = nn.Linear(feature_dim + hiddenNum, outputDim)

    def forward(self, x, video_feature: bool = False):
        # x shape (bs, seq_len, input_dim)
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        # fcOutput = self.fc(hn)
        rnnOutput, hn = self.cell(x, (h0, c0))
        hn = hn[0][-1].view(batchSize, 1, self.hiddenNum)  # video level feature
        hns = torch.tile(hn, (1, x.size(1), 1))  # (bs, seq_len, hiddenNum)
        x = self.input2feature(x)  # (bs, seq_len, feature_dim)
        aggregation = torch.cat((x, hns), dim=2)  # (bs, seq_len, hiddenNum + feature_dim)
        alphas = self.alpha(x)  # (bs, seq_len, 1) frame level attention weights
        betas = self.beta(aggregation)  # (bs, seq_len, 1)
        alpha_betas = torch.mul(alphas, betas)  # (bs, seq_len, 1)
        weighted_sum = torch.sum(torch.mul(alpha_betas, aggregation), dim=1)  # (bs, hiddenNum + feature_dim)
        features = weighted_sum / torch.sum(alpha_betas, dim=1)  # (bs, hiddenNum + inputDim)
        output = self.lastL(features)
        return output
class ORILSTM(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda):
        super(ORILSTM, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda)
    def forward(self, x):

        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        rnnOutput, hn = self.cell(x, (h0, c0))
        hn = hn[0].view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput
class FollowLSTM(BaseModel):
    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda):
        super(FollowLSTM, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell, use_cuda)
        self.alpha = nn.Sequential(nn.Linear(512,1),nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(1024,1),nn.Sigmoid())
    def forward(self, x ):
        alphas = self.alpha(x)
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        rnnOutput, hn = self.cell(x, (h0, c0))
        hn = hn[0].view(batchSize, self.hiddenNum)
        hx = torch.cat((x,hn[:,None,:].repeat(1,50,1)),dim=2)
        betas = self.beta(hx)
        hx = torch.mul(hx,alphas*betas).sum(1)/torch.sum(alphas*betas,dim=1)
        fcOutput = self.fc(hx)
        return fcOutput
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_features = 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x=''):
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)

        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)

        f = f.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]

        return f
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock_AT(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_AT, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ResNet_AT(nn.Module):
    def __init__(self, block, layers, num_classes=1000, end2end=True, at_type=''):
        self.inplanes = 64
        self.end2end = end2end
        super(ResNet_AT, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.6)
        self.alpha = nn.Sequential(nn.Linear(512, 1),
                                   nn.Sigmoid())

        self.beta = nn.Sequential(nn.Linear(1024, 1),
                                  nn.Sigmoid())

        self.pred_fc1 = nn.Linear(512, 7)
        self.pred_fc2 = nn.Linear(1024, 7)
        self.at_type = at_type

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x='', phrase='train', AT_level='first_level', vectors='', vm='', alphas_from1='',
                index_matrix=''):

        vs = []
        alphas = []

        assert phrase == 'train' or phrase == 'eval'
        assert AT_level == 'first_level' or AT_level == 'second_level' or AT_level == 'pred'
        if phrase == 'train':
            num_pair = 3

            for i in range(num_pair):
                f = x[:, :, :, :, i]  # x[128,3,224,224]

                f = self.conv1(f)
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)

                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)

                f = f.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]

                # MN_MODEL(first Level)
                vs.append(f)
                alphas.append(self.alpha(self.dropout(f)))

            vs_stack = torch.stack(vs, dim=2)
            alphas_stack = torch.stack(alphas, dim=2)

            if self.at_type == 'self-attention':
                vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
            if self.at_type == 'self_relation-attention':
                vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
                betas = []
                for i in range(len(vs)):
                    vs[i] = torch.cat([vs[i], vm1], dim=1)
                    betas.append(self.beta(self.dropout(vs[i])))

                cascadeVs_stack = torch.stack(vs, dim=2)
                betas_stack = torch.stack(betas, dim=2)
                output = cascadeVs_stack.mul(betas_stack * alphas_stack).sum(2).div((betas_stack * alphas_stack).sum(2))

            if self.at_type == 'self-attention':
                vm1 = self.dropout(vm1)
                pred_score = self.pred_fc1(vm1)

            if self.at_type == 'self_relation-attention':
                output = self.dropout2(output)
                pred_score = self.pred_fc2(output)

            return pred_score

        if phrase == 'eval':
            if AT_level == 'first_level':
                f = self.conv1(x)
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)

                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)

                f = f.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]
                # MN_MODEL(first Level)
                alphas = self.alpha(self.dropout(f))

                return f, alphas

            if AT_level == 'second_level':
                assert self.at_type == 'self_relation-attention'
                vms = index_matrix.permute(1, 0).mm(vm)  # [381, 21783] -> [21783,381] * [381,512] --> [21783, 512]
                vs_cate = torch.cat([vectors, vms], dim=1)
                return vs_cate

            if AT_level == 'pred':
                if self.at_type == 'self-attention':
                    pred_score = self.pred_fc1(self.dropout(vm))

                return pred_score
''' self-attention; relation-attention '''
def resnet18_at(pretrained=False, **kwargs):
    # Constructs base a ResNet-18 model.
    model = ResNet_AT(BasicBlock_AT, [2, 2, 2, 2], **kwargs)
    return model
class EmoTransformer(nn.Module):
    def __init__(self, input, nhead, num_layers, batch_first, output_dim):  # bs first True
        '''
            1. embedding layer is may be needed, because more parameters has more fitting capacity.
            2. position encoding layer has one more position for cls token, thus, concatenation cls token to input follows embedding.
            3. transformer does not care seq_len.
            4. after encoder, cls token will be used for classification.
            5. encoder has normalization layer, input data normalizing is no needed?
        '''
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        from dataProcess import config
        super(EmoTransformer, self).__init__()
        d_model = config['T_proj_dim']
        self.input_embedding = nn.Sequential(nn.Linear(input, d_model))
        # pe = torch.zeros(25 + 1, d_model)  # add cls token
        # position = torch.arange(0, 26).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)  # sin
        # pe[:, 1::2] = torch.cos(position * div_term)  # cos
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # self.pe = pe.unsqueeze(0)
        self.pos_embedding = nn.Parameter(torch.randn(1, config['window_size'] + 1, d_model))
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=batch_first,
                                                     dim_feedforward=config['T_forward_dim'],
                                                     activation=config['T_activation'])
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.pred = nn.Sequential(nn.Linear(d_model, output_dim))
        self.pred1 = nn.Sequential(nn.Linear(d_model, output_dim))
    def forward(self, x, att_mask: bool = False, phrase='train', v_s=''):
        if phrase == 'train':
            tem_m = torch.sum(x, dim=2)  # (bs, seq_len)
            x = self.input_embedding(x)  # (bs,seq_len,d_model)
            cls_tokens = torch.tile(self.cls_token, (x.shape[0], 1, 1))  # (bs,1,d_model)
            cls_x = torch.cat((cls_tokens, x), dim=1)  # (bs,seq_len + 1,d_model)
            # make a key padding mask matrix
            mask_m = tem_m == 0  # (bs, seq_len)
            cls_m = torch.zeros((x.shape[0], 1), dtype=torch.bool).cuda()
            mask_m = torch.cat((cls_m, mask_m), dim=1)  # (bs, seq_len+1)
            cls_x += self.pos_embedding
            # cls_x = cls_x + Variable(self.pe[:, :cls_x.size(1)],requires_grad=False).cuda() # (bs,seq_len + 1,d_model)
            if att_mask:
                x_ = self.transformer_encoder(cls_x, src_key_padding_mask=mask_m)  # (bs,seq_len + 1,d_model)
                self.transformer_encoder.layers
            else:
                x_ = self.transformer_encoder(cls_x)
            v_feature = x_[:, 0, :]  # (bs,1,d_model)
            return self.pred1(v_feature)
        if phrase == 'mlpIn':
            output = self.pred(v_s) # (video, 7)
            return output
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.AE_encoder = nn.Sequential(nn.Linear(204, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(),
                                        nn.Linear(32, config['AE_mid_dim']))
        self.AE_decoder = nn.Sequential(nn.Linear(config['AE_mid_dim'], 32), nn.ReLU(), nn.Linear(32, 128), nn.ReLU(),
                                        nn.Linear(128, 204))

    def forward(self, x, use_mid: bool = False):
        mid = self.AE_encoder(x)
        if use_mid:
            return mid
        output = self.AE_decoder(mid)
        return output
class ViLT(nn.Module):
    def __init__(self, model_type_num: int, lms_type_dim: int, rgb_type_dim: int, d_model: int, nhead: int,
                 forward_dim: int, block_num: int, batch_fist: bool):
        super(ViLT, self).__init__()
        self.lms_type_project_layer = nn.Linear(lms_type_dim, d_model)  # (204, embedding_dim)
        self.rgb_type_project_layer = nn.Linear(rgb_type_dim, d_model)  # (p*p*c, embedding_dim)
        self.model_emd = nn.Embedding(model_type_num, d_model)  # (model_type_num, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.second_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.lms_pos_emb = nn.Parameter(torch.randn(1, config['window_size'] + 1, d_model))
        self.rgb_pos_emb = nn.Parameter(torch.randn(1, config['window_size'] + 1, d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=forward_dim,
                                                        activation=config['ViLT_activation'], batch_first=batch_fist)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, block_num)
        self.MLP = nn.Sequential(nn.Linear(config['ViLT_embedding_dim'], config['ViLT_embedding_dim']), nn.Tanh(),
                                 nn.Linear(config['ViLT_embedding_dim'], 7))  # pool + classification

    def forward(self, x):
        x_rgb, x_lms = torch.split(x, 512, dim=2)
        lms_emb = self.lms_type_project_layer(x_lms)
        rgb_emb = self.rgb_type_project_layer(x_rgb)
        input_emb = torch.cat((torch.tile(self.cls_token, dims=(x.shape[0], 1, 1)), lms_emb, torch.tile(self.second_token, dims=(x.shape[0], 1, 1)), rgb_emb), dim=1)
        input_emb += torch.cat((self.lms_pos_emb, self.rgb_pos_emb), dim=1)
        output_emb = self.encoder(input_emb)
        pred = self.MLP(output_emb[:, 0, :])
        return pred
class MultiEmoTransformer(nn.Module):
    def __init__(self,lms3dpro, rgbpro, input, nhead, num_layers, batch_first, output_dim):  # bs first True
        '''
            1. embedding layer is may be needed, because more parameters has more fitting capacity.
            2. position encoding layer has one more position for cls token, thus, concatenation cls token to input follows embedding.
            3. transformer does not care seq_len.
            4. after encoder, cls token will be used for classification.
            5. encoder has normalization layer, input data normalizing is no needed?
        '''
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        from dataProcess import config
        super(MultiEmoTransformer, self).__init__()
        d_model = config['T_proj_dim']
        self.lms3d_embedding = nn.Sequential(nn.Linear(config['T_lms3d_dim'], config['T_pro_dim']))
        self.rgb_embedding = nn.Sequential(nn.Linear(config['T_rgb_dim'], config['T_pro_dim']))
        self.input_embedding = nn.Sequential(nn.Linear(config['T_pro_dim']*2, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, config['window_size'] + 1, d_model))
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=batch_first,
                                                     dim_feedforward=config['T_forward_dim'],
                                                     activation=config['T_activation'])
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.pred = nn.Sequential(nn.Linear(d_model, output_dim))
    def forward(self, lms3d, rgb, att_mask: bool = False):
        l = self.lms3d_embedding(lms3d)
        r = self.rgb_embedding(rgb)
        x = torch.cat((l,r),dim=2)
        tem_m = torch.sum(x, dim=2)  # (bs, seq_len)
        x = self.input_embedding(x)  # (bs,seq_len,d_model)
        cls_tokens = torch.tile(self.cls_token, (x.shape[0], 1, 1))  # (bs,1,d_model)
        cls_x = torch.cat((cls_tokens, x), dim=1)  # (bs,seq_len + 1,d_model)
        # make a key padding mask matrix
        mask_m = tem_m == 0  # (bs, seq_len)
        cls_m = torch.zeros((x.shape[0], 1), dtype=torch.bool).cuda()
        mask_m = torch.cat((cls_m, mask_m), dim=1)  # (bs, seq_len+1)
        cls_x += self.pos_embedding
        # cls_x = cls_x + Variable(self.pe[:, :cls_x.size(1)],requires_grad=False).cuda() # (bs,seq_len + 1,d_model)
        if att_mask:
            x_ = self.transformer_encoder(cls_x, src_key_padding_mask=mask_m)  # (bs,seq_len + 1,d_model)
            # self.transformer_encoder.layers
        else:
            x_ = self.transformer_encoder(cls_x)
        v_feature = x_[:, 0, :]  # (bs,1,d_model)
        return self.pred(v_feature)
from utils.config import cfg
class MAEEncoder(nn.Module):
    def __init__(self, embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        from timm.models.vision_transformer import Block
        # --------------------------------------------------------------------------
        self.encoder_embed = nn.Linear(cfg.input_len, embed_dim)  # bias = True
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1)
            for i in range(depth)])
        self.pred = nn.Linear(embed_dim, cfg.train_out_dim)
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
    def initialize_weights(self):
        import numpy as np
        from util.pos_embed import get_1d_sincos_pos_embed_from_grid
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(cfg.seq_len + 1))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        # with torch.no_grad():
        x = self.encoder_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]  # 传播算法

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        return self.pred(x[:,0,:])
class NormalB16ViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self,weights):
        # super().__init__(embed_dim=1280,num_heads=16,patch_size=14) # h-14的设置
        super().__init__()
        if weights is not None:
            self.load_from(weights)
        else:
            pass
        # self.lms_porj = nn.Linear(204, 256)
        self.emb_proj = nn.Linear(config['T_input_dim'],self.embed_dim) # 1024->768
        self.pos_embed_my = nn.Parameter(torch.zeros(1, config['window_size']+1, self.embed_dim))
        # self.pos_embed_my.data.copy_(self.pos_embed[:,:config['window_size']+1,:]) # cut position embedding
        self.pred = nn.Linear(self.embed_dim,config['T_output_dim'])
        pass

    @torch.no_grad()
    def load_from(self, weights):
        def _n2p(w, t=True):
            if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
                w = w.flatten()
            if t:
                if w.ndim == 4:
                    w = w.transpose([3, 2, 0, 1])
                elif w.ndim == 3:
                    w = w.transpose([2, 0, 1])
                elif w.ndim == 2:
                    w = w.transpose([1, 0])
            return torch.from_numpy(w)
        self.pos_embed.data.copy_(torch.from_numpy(weights['Transformer/posembed_input/pos_embedding']))
        self.cls_token.data.copy_(torch.from_numpy(weights['cls']))
        for l,b in enumerate(self.blocks):
            ################ln weight######################
            b.norm1.bias.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_0/bias']))
            b.norm1.weight.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_0/scale']))
            b.norm2.bias.data.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_2/bias']))
            b.norm2.weight.data.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_2/scale']))
            ###############mlp weight######################
            b.mlp.fc1.weight.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_0/kernel']).t())
            b.mlp.fc1.bias.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_0/bias']).t())
            b.mlp.fc2.weight.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_1/kernel']).t())
            b.mlp.fc2.bias.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_1/bias']).t())
            ###############attention weight################
            b.attn.qkv.weight.copy_(torch.cat([_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
            b.attn.qkv.bias.copy_(torch.cat([_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
            b.attn.proj.weight.copy_(_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/out/kernel']).flatten(1))
            b.attn.proj.bias.copy_(_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/out/bias']))
    @torch.no_grad()
    def SinusoidalEncoding(self, seq_len, d_model):
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            for pos in range(seq_len)])
        pos_table[0, 0::2] = np.sin(pos_table[0, 0::2])
        pos_table[0, 1::2] = np.cos(pos_table[0, 1::2])
        return torch.FloatTensor(pos_table)

    def forward(self, x, position_embeddings=None):
        # x_l = x[:,:,:204]
        # x_r = x[:,:,204:]
        # x_l = self.lms_porj(x_l)
        # x = torch.cat((x_l,x_r),dim=2)
        x = self.emb_proj(x)
        if position_embeddings is None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x),dim=1) + self.pos_embed_my
        else:
            cls_token_pe = torch.tile(self.SinusoidalEncoding(1,self.embed_dim),(x.shape[0],1)).cuda()
            position_embeddings = torch.from_numpy(position_embeddings).cuda()
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + torch.cat((cls_token_pe[:, None, :], position_embeddings.expand(x.shape[0], -1, -1)),dim=1)
        #x(b,s,d)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        pred = self.pred(x[:,0,:])
        return pred
class B16ViT_AB(timm.models.vision_transformer.VisionTransformer):
    def __init__(self,weights):
        # super().__init__(embed_dim=1280,num_heads=16,patch_size=14) # h-14的设置
        super().__init__()
        if weights is not None:
            self.load_from(weights)
        else:
            pass
        self.sig = nn.Sigmoid()
        self.beta = nn.Sequential(nn.Linear(config['T_input_dim']*2,1),nn.Sigmoid())
        self.emb_proj = nn.Linear(config['T_input_dim'],self.embed_dim) # 1024->768
        self.pos_embed_my = nn.Parameter(torch.zeros(1, config['window_size']+1, self.embed_dim))
        self.pred = nn.Linear(self.embed_dim,config['T_output_dim'])
        pass
    @torch.no_grad()
    def load_from(self, weights):
        def _n2p(w, t=True):
            if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
                w = w.flatten()
            if t:
                if w.ndim == 4:
                    w = w.transpose([3, 2, 0, 1])
                elif w.ndim == 3:
                    w = w.transpose([2, 0, 1])
                elif w.ndim == 2:
                    w = w.transpose([1, 0])
            return torch.from_numpy(w)
        self.pos_embed.data.copy_(torch.from_numpy(weights['Transformer/posembed_input/pos_embedding']))
        self.cls_token.data.copy_(torch.from_numpy(weights['cls']))
        for l,b in enumerate(self.blocks):
            ################ln weight######################
            b.norm1.bias.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_0/bias']))
            b.norm1.weight.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_0/scale']))
            b.norm2.bias.data.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_2/bias']))
            b.norm2.weight.data.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_2/scale']))
            ###############mlp weight######################
            b.mlp.fc1.weight.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_0/kernel']).t())
            b.mlp.fc1.bias.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_0/bias']).t())
            b.mlp.fc2.weight.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_1/kernel']).t())
            b.mlp.fc2.bias.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_1/bias']).t())
            ###############attention weight################
            b.attn.qkv.weight.copy_(torch.cat([_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
            b.attn.qkv.bias.copy_(torch.cat([_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
            b.attn.proj.weight.copy_(_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/out/kernel']).flatten(1))
            b.attn.proj.bias.copy_(_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/out/bias']))
    @torch.no_grad()
    def SinusoidalEncoding(self, seq_len, d_model):
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            for pos in range(seq_len)])
        pos_table[0, 0::2] = np.sin(pos_table[0, 0::2])
        pos_table[0, 1::2] = np.cos(pos_table[0, 1::2])
        return torch.FloatTensor(pos_table)
    def forward(self, rgb_fea, position_embeddings=None):
        x0 = self.emb_proj(rgb_fea)
        alphas = self.sig(x0)
        if position_embeddings is None:
            x = torch.cat((self.cls_token.expand(x0.shape[0], -1, -1), x0),dim=1) + self.pos_embed_my
        else:
            cls_token_pe = torch.tile(self.SinusoidalEncoding(1,self.embed_dim),(x0.shape[0],1)).cuda()
            position_embeddings = torch.from_numpy(position_embeddings).cuda()
            x = torch.cat((self.cls_token.expand(x0.shape[0], -1, -1), x0), dim=1) + torch.cat((cls_token_pe[:, None, :], position_embeddings.expand(x0.shape[0], -1, -1)),dim=1)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        v_fea = x[:,0,:]
        sum_fea = torch.cat((x0, v_fea.repeat(1, 50, 1)), dim=2)
        betas = self.beta(sum_fea)
        hx = torch.mul(sum_fea, alphas * betas).sum(1) / torch.sum(alphas * betas, dim=1)
        pred = self.pred(hx)
        return pred
class B16ViT_AB_Res(timm.models.vision_transformer.VisionTransformer):
    def __init__(self,block,layers):
        ########################################resnet18 参数##########################
        self.inplanes = 64
        super(B16ViT_AB_Res, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        ########################################vit 参数###############################
        self.emb_proj = nn.Linear(config['T_input_dim'],self.embed_dim) # 1024->768
        self.alpha = nn.Sequential(nn.Linear(self.embed_dim,1),nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(config['T_proj_dim']*2,1),nn.Sigmoid())
        self.pos_embed_my = nn.Parameter(torch.zeros(1, config['window_size']+1, self.embed_dim))
        self.pred = nn.Linear(self.embed_dim*2,config['T_output_dim'])
        self.blank = torch.zeros((1,512)).cuda()
        pass
    @torch.no_grad()
    def load_from(self, weights):
        def _n2p(w, t=True):
            if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
                w = w.flatten()
            if t:
                if w.ndim == 4:
                    w = w.transpose([3, 2, 0, 1])
                elif w.ndim == 3:
                    w = w.transpose([2, 0, 1])
                elif w.ndim == 2:
                    w = w.transpose([1, 0])
            return torch.from_numpy(w)
        self.pos_embed.data.copy_(torch.from_numpy(weights['Transformer/posembed_input/pos_embedding']))
        self.cls_token.data.copy_(torch.from_numpy(weights['cls']))
        for l,b in enumerate(self.blocks):
            ################ln weight######################
            b.norm1.bias.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_0/bias']))
            b.norm1.weight.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_0/scale']))
            b.norm2.bias.data.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_2/bias']))
            b.norm2.weight.data.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/LayerNorm_2/scale']))
            ###############mlp weight######################
            b.mlp.fc1.weight.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_0/kernel']).t())
            b.mlp.fc1.bias.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_0/bias']).t())
            b.mlp.fc2.weight.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_1/kernel']).t())
            b.mlp.fc2.bias.copy_(torch.from_numpy(weights[f'Transformer/encoderblock_{l}/MlpBlock_3/Dense_1/bias']).t())
            ###############attention weight################
            b.attn.qkv.weight.copy_(torch.cat([_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
            b.attn.qkv.bias.copy_(torch.cat([_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
            b.attn.proj.weight.copy_(_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/out/kernel']).flatten(1))
            b.attn.proj.bias.copy_(_n2p(weights[f'Transformer/encoderblock_{l}/MultiHeadDotProductAttention_1/out/bias']))
    @torch.no_grad()
    def SinusoidalEncoding(self, seq_len, d_model):
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            for pos in range(seq_len)])
        pos_table[0, 0::2] = np.sin(pos_table[0, 0::2])
        pos_table[0, 1::2] = np.cos(pos_table[0, 1::2])
        return torch.FloatTensor(pos_table)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forwardRes(self, x):
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)

        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)

        f = f.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]
        return f
    def forwardVit(self, rgb, position_embeddings=None):
        x0 = self.emb_proj(rgb)
        alphas = self.alpha(x0)
        with torch.no_grad():
            if position_embeddings is None:
                x = torch.cat((self.cls_token.expand(x0.shape[0], -1, -1), x0), dim=1) + self.pos_embed_my
            else:
                cls_token_pe = torch.tile(self.SinusoidalEncoding(1, self.embed_dim), (x0.shape[0], 1)).cuda()
                position_embeddings = torch.from_numpy(position_embeddings).cuda()
                x = torch.cat((self.cls_token.expand(x0.shape[0], -1, -1), x0), dim=1) + torch.cat(
                    (cls_token_pe[:, None, :], position_embeddings.expand(x0.shape[0], -1, -1)), dim=1)
            x = self.norm_pre(x)
            x = self.blocks(x)
            x = self.norm(x)
            v_fea = x[:, 0, :]
            sum_fea = torch.cat((x0, v_fea[:,None,:].repeat(1, 270, 1)), dim=2)
        betas = self.beta(sum_fea)
        hx = torch.mul(sum_fea, alphas * betas).sum(1) / torch.sum(alphas * betas, dim=1)
        pred = self.pred(hx)
        return pred
    def forward(self,x,pe=None):
        vit_x = torch.zeros((0,270,512)).cuda()
        for bs_unit in range(x.shape[0]):
            real_seq = x[bs_unit]
            real_seq_fea = self.forwardRes(real_seq)
            real_seq_len = real_seq_fea.shape[0]
            EOS_num = 270-real_seq_len
            blanks = torch.tile(self.blank,(EOS_num,1))
            sup_seq = torch.cat((real_seq_fea, blanks),dim=0)
            vit_x = torch.cat((vit_x, sup_seq[None,:,:]),dim=0)
        return self.forwardVit(vit_x,position_embeddings=pe)
if __name__ == '__main__':
    import dataProcess
    cnn_prh = '/home/exp-10086/Project/Emotion-FAN-master/pretrain_model/Resnet18_FER+_pytorch.pth.tar'
    vit_pth = '/home/exp-10086/Project/ferdataset/1test_37.pkl'
    cnn_check = torch.load(cnn_prh)['state_dict']
    vit_check = torch.load(vit_pth)['state_dict']
    hybrid = B16ViT_AB_Res(BasicBlock,[2,2,2,2])
    model_state = hybrid.state_dict()
    for key in cnn_check:
        for key in cnn_check:
            if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
                continue
            else:
                model_state[key.replace('module.', '')] = cnn_check[key]
    for key in vit_check:
        model_state[key] = vit_check[key]
    imgin = torch.randn(size=(16,50,3,224,224)).cuda()
    hybrid.train()
    hybrid.cuda()
    pos_encoding = dataProcess.Utils.SinusoidalEncoding(270, config['T_proj_dim'])
    y = hybrid(imgin,pe=pos_encoding)
    pass
