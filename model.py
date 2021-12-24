
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class CONV4d(nn.Module):
    def __init__(self, n_in, n_out, t_size = 3 ,s_size = 3, t_stride = 1,s_stride = 1,s_padding = 1):
        super(CONV4d, self).__init__()
    
        self.n_in = n_in
        self.n_out = n_out
        self.s_size = s_size
        self.s_stride = s_stride
        self.s_padding = s_padding
        self.t_size = t_size
        self.t_stride = t_stride
        self.conv3d_layers = torch.nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        for i in range(t_size):
            conv3d_layer = torch.nn.Conv3d(in_channels=self.n_in,
                                           out_channels=self.n_out,
                                           kernel_size=self.s_size,
                                           stride = self.s_stride,
                                           padding=self.s_padding)
            self.conv3d_layers.append(conv3d_layer)
            
    def forward(self, x):  
        (b,c,l,h,w,d) = x.shape
        stack_4d = [ None ]*(int((l-self.t_size)/self.t_stride)+1)
        for i in range(self.t_size):
            for j in range(0,l-self.t_size+1,self.t_stride):
                input_x = torch.reshape(x[:,:,self.t_size-i-1+j,:],(b,c,h,w,d))
                out_x = self.conv3d_layers[i](input_x)
                stack_check = ((stack_4d[int(j/self.t_stride)]))
                if stack_check is None:
                    stack_4d[int(j/self.t_stride)] = out_x
                else:
                    stack_4d[int(j/self.t_stride)] = stack_4d[int(j/self.t_stride)] + out_x
        out = torch.stack(stack_4d,2)
        (b,c,l,h,w,d) = out.shape
        out =  torch.reshape(out,(b,c*l,h,w,d))
        out = self.relu(out)
        return out 

class Attention_network(nn.Module):
    def __init__(self,conv4d = CONV4d):
        super(Attention_network, self).__init__()
        
        self.conv4d = conv4d(1, 3, t_size = 3 , s_size = 3, t_stride = 2,s_stride = 2,s_padding=1) 
        self.residual_block1 = ResidualBlock(21, 16) 
        self.attention_module1 = AttentionModule_stage1(16, 16)
        self.residual_block2 = ResidualBlock(16, 32, 2)
        self.attention_module2 = AttentionModule_stage2(32, 32)
        self.residual_block3 = ResidualBlock(32, 64, 2) 
        self.attention_module3 = AttentionModule_stage3(64, 64)
        self.residual_block4 = ResidualBlock(64, 128, 2) 
        self.attention_module4 = AttentionModule_stage4(128, 128)
        self.pooling = nn.Sequential(
            nn.AvgPool3d(kernel_size=(5,6,6), stride=1))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128,7))
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv4d(x)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.attention_module4(out)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.stride = stride
        self.conv1 = nn.Conv3d(int(input_channels), int(output_channels), 3, stride,  padding = 1,bias = False)
        self.bn1 = nn.BatchNorm3d(int(output_channels))
        self.conv3 = nn.Conv3d(int(output_channels), output_channels, 3, 1, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            self.conv4 = nn.Sequential(
                nn.Conv3d(int(input_channels), output_channels , 1, stride, bias = False),
                nn.BatchNorm3d(output_channels))
        else:
            self.conv4 = None
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        
        
    def forward(self, x):
        residual = x
        if self.conv4 is not None:
            residual = self.conv4(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class AttentionModule_stage1(nn.Module):
    def __init__(self, in_channels, out_channels, size1=(40,48,44), size2=(20,24,22), size3=(10,12,11)):
        super(AttentionModule_stage1, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )
        self.mpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)
        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        self.softmax4_blocks = ResidualBlock(in_channels, out_channels)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
        self.softmax6_blocks = nn.Sequential(
            nn.Conv3d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm3d(out_channels),
           nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_channels, out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_interp3 = nn.functional.interpolate(out_softmax3,size=self.size3,mode = 'trilinear') + out_softmax2
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = nn.functional.interpolate(out_softmax4,size=self.size2,mode = 'trilinear') + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = nn.functional.interpolate(out_softmax5,size=self.size1,mode = 'trilinear') + out_trunk
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax6) * out_trunk
        out_last = self.last_blocks(out)
        return out_last
    
    
class AttentionModule_stage2(nn.Module):
    def __init__(self, in_channels, out_channels, size1=(20, 24,22), size2=(10,12,11)):
        super(AttentionModule_stage2, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )
        self.mpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
        self.softmax4_blocks = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out_trunk = self.trunk_branches(x)   
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_interp2 = nn.functional.interpolate(out_softmax2,size=self.size2,mode = 'trilinear') + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = nn.functional.interpolate(out_softmax3,size=self.size1,mode = 'trilinear') + out_trunk
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax4) * out_trunk
        out_last = self.last_blocks(out)
        return out_last


class AttentionModule_stage3(nn.Module):
    # input image size is 14*14
    def __init__(self, in_channels, out_channels, size1=(10,12,11)):
        super(AttentionModule_stage3, self).__init__()
        self.size1 = size1
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )
        self.mpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        self.softmax2_blocks = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_channels, out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_interp1 = nn.functional.interpolate(out_softmax1,size=self.size1,mode = 'trilinear') + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)
        return out_last
    
class AttentionModule_stage4(nn.Module):
    def __init__(self, in_channels, out_channels, size1=(5,6,5)):
        super(AttentionModule_stage4, self).__init__()
        self.size1 = size1
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )
        self.mpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        self.softmax2_blocks = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_channels, out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out_trunk = self.trunk_branches(x)
        out_softmax1 = self.softmax1_blocks(x)
        out_interp1 = out_softmax1 + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)
        return out_last

