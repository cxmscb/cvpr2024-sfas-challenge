import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet import EfficientNet


class SWL(nn.Module):
    def __init__(self, num_channels=30):
        super(SWL, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels // 8, out_channels=num_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        B, C, _, _ = x.size()
        x_avg, x_max = self.avgpool(x), self.maxpool(x)
        avg_weights, max_weights = self.channel_attention(x_avg), self.channel_attention(x_max)
        band_weights = self.sigmoid(avg_weights + max_weights)
        return band_weights



class Texture_Enhance_v2(nn.Module):
    def __init__(self,num_features,num_attentions):
        super().__init__()
        self.output_features=num_features
        self.output_features_d=num_features
        self.conv_extract=nn.Conv2d(num_features,num_features,3,padding=1)
        self.conv0=nn.Conv2d(num_features*num_attentions,num_features*num_attentions,5,padding=2,groups=num_attentions)
        self.conv1=nn.Conv2d(num_features*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn1=nn.BatchNorm2d(num_features*num_attentions)
        self.conv2=nn.Conv2d(num_features*2*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn2=nn.BatchNorm2d(2*num_features*num_attentions)
        self.conv3=nn.Conv2d(num_features*3*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn3=nn.BatchNorm2d(3*num_features*num_attentions)
        self.conv_last=nn.Conv2d(num_features*4*num_attentions,num_features*num_attentions,1,groups=num_attentions)
        self.bn4=nn.BatchNorm2d(4*num_features*num_attentions)
        self.bn_last=nn.BatchNorm2d(num_features*num_attentions)
        
        self.M=num_attentions
    def cat(self,a,b):
        B,C,H,W=a.shape
        c=torch.cat([a.reshape(B,self.M,-1,H,W),b.reshape(B,self.M,-1,H,W)],dim=2).reshape(B,-1,H,W)
        return c

    def forward(self,feature_maps,attention_maps=(1,1)):
        B,N,H,W=feature_maps.shape
        if type(attention_maps)==tuple:
            attention_size=(int(H*attention_maps[0]),int(W*attention_maps[1]))
        else:
            attention_size=(attention_maps.shape[2],attention_maps.shape[3])
        feature_maps=self.conv_extract(feature_maps)
        # feature_maps_d=F.adaptive_avg_pool2d(feature_maps,attention_size)
        feature_maps_d = F.avg_pool2d(feature_maps,(4,4))
        # print(feature_maps.shape, feature_maps_d.shape, attention_size)
        if feature_maps.size(2)>feature_maps_d.size(2):
            feature_maps=feature_maps-F.interpolate(feature_maps_d,(feature_maps.shape[2],feature_maps.shape[3]),mode='nearest')
        attention_maps=(torch.tanh(F.interpolate(attention_maps.detach(),(H,W),mode='bilinear',align_corners=True))).unsqueeze(2) if type(attention_maps)!=tuple else 1
        feature_maps=feature_maps.unsqueeze(1)
        feature_maps=(feature_maps*attention_maps).reshape(B,-1,H,W)
        feature_maps0=self.conv0(feature_maps)
        feature_maps1=self.conv1(F.relu(self.bn1(feature_maps0),inplace=True))
        feature_maps1_=self.cat(feature_maps0,feature_maps1)
        feature_maps2=self.conv2(F.relu(self.bn2(feature_maps1_),inplace=True))
        feature_maps2_=self.cat(feature_maps1_,feature_maps2)
        feature_maps3=self.conv3(F.relu(self.bn3(feature_maps2_),inplace=True))
        feature_maps3_=self.cat(feature_maps2_,feature_maps3)
        feature_maps=F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_),inplace=True))),inplace=True)
        feature_maps=feature_maps.reshape(B,-1,N,H,W)
        return feature_maps,feature_maps_d


class AttentionMap(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        self.register_buffer('mask',torch.zeros([1,1,24,24]))
        self.mask[0,0,2:-2,2:-2]=1
        self.num_attentions=out_channels
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1) #extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.num_attentions==0:
            return torch.ones([x.shape[0],1,1,1],device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)+1
        mask=F.interpolate(self.mask,(x.shape[2],x.shape[3]),mode='nearest')
        return x*mask


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, features, attentions,norm=2):
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions=F.interpolate(attentions,size=(H,W), mode='bilinear', align_corners=True)
        if norm==1:
            attentions=attentions+1e-8
        if len(features.shape)==4:
            feature_matrix=torch.einsum('imjk,injk->imn', attentions, features)
        else:
            feature_matrix=torch.einsum('imjk,imnjk->imn', attentions, features)
        if norm==1:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)
            feature_matrix/=w
        if norm==2:
            feature_matrix = F.normalize(feature_matrix,p=2,dim=-1)
        if norm==3:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)+1e-8
            feature_matrix/=w
        return feature_matrix


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out


class MAT(nn.Module):
    def __init__(self, net='efficientnet-b3',feature_layer='b3',attention_layer='final',num_classes=1, M=8,mid_dims=256,\
    dropout_rate=0.5,drop_final_rate=0.5, pretrained=False,alpha=0.05,size=(256,256),margin=1,inner_margin=[0.01,0.02]):
        super(MAT, self).__init__()
        self.num_classes = num_classes
        self.swl = SWL(33)
        self.M = M
        if 'xception' in net:
            self.net=xception(num_classes)
        elif net.split('-')[0]=='efficientnet':
            self.net=EfficientNet.from_pretrained(net,advprop=True, num_classes=num_classes)
        self.feature_layer=feature_layer
        self.attention_layer=attention_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1,33,size[0],size[1]))
        num_features=layers[self.feature_layer].shape[1]
        self.mid_dims=mid_dims
        if pretrained:
            a=torch.load(pretrained,map_location='cpu')
            keys={i:a['state_dict'][i] for i in a.keys() if i.startswith('net')}
            if not keys:
                keys=a['state_dict']
            self.net.load_state_dict(keys,strict=False)
        self.attentions = AttentionMap(layers[self.attention_layer].shape[1], self.M)
        self.atp=AttentionPooling()
        self.texture_enhance=Texture_Enhance_v2(num_features,M)
        self.num_features=self.texture_enhance.output_features
        self.num_features_d=self.texture_enhance.output_features_d
        self.projection_local=nn.Sequential(nn.Linear(M*self.num_features,mid_dims),nn.BatchNorm1d(mid_dims),nn.Mish(),nn.Linear(mid_dims,mid_dims),nn.BatchNorm1d(mid_dims))
        self.project_final=nn.Sequential(nn.Linear(layers['final'].shape[1],mid_dims),nn.BatchNorm1d(mid_dims))
        self.ensemble_classifier_fc=nn.Sequential(nn.Linear(mid_dims*2,mid_dims),nn.BatchNorm1d(mid_dims),nn.Mish(),nn.Linear(mid_dims,num_classes))
        self.head_mlp = nn.Sequential(nn.Linear(mid_dims*2, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Linear(1024, 1024))
        self.dropout=nn.Dropout2d(dropout_rate,inplace=True)
        del layers
        layers = None

    def forward(self, x):
        band_weights = self.swl(x)
        x = x + x * band_weights
        layers = self.net(x)
        raw_attentions = layers[self.attention_layer]
        attention_maps=self.attentions(raw_attentions)
        feature_maps = layers[self.feature_layer]
        feature_maps,feature_maps_d=self.texture_enhance(feature_maps,attention_maps)
        feature_matrix=self.atp(feature_maps,attention_maps)
        B,M,N = feature_matrix.size()
        feature_matrix=self.dropout(feature_matrix)
        feature_matrix=feature_matrix.view(B,-1)
        feature_matrix=F.mish(self.projection_local(feature_matrix))
        final=layers['final']
        attention_maps2=attention_maps.sum(dim=1,keepdim=True)
        final=self.atp(final,attention_maps2,norm=1).squeeze(1)
        projected_final=F.mish(self.project_final(final))
        feature_matrix=torch.cat((feature_matrix,projected_final),1)
        ensemble_logit=self.ensemble_classifier_fc(feature_matrix)
        # return ensemble_logit, feature_matrix
        return ensemble_logit, F.normalize(self.head_mlp(feature_matrix), dim=1)


if __name__ == "__main__":
    model = MAT()
    pass