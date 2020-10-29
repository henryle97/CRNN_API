import torch
import torch.nn as nn
import torch.nn.functional as F

from CRNN_TEXTREG.model.layer import conv_bn_relu
# from efficientnet_pytorch import EfficientNet


class BidirectionalLSTM(nn.Module):
    def __init__(self, in_channels, number_hidden, out_channels):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(in_channels, number_hidden, bidirectional=True)
        self.embedding = nn.Linear(number_hidden * 2, out_channels)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0  # replace all nan/inf in gradients to zero


# class CRNN(nn.Module):
#     def __init__(self, n_channels, number_class, number_hidden):
#         super(CRNN, self).__init__()
#         #0
#         self.conv1 = conv_bn_relu(n_channels, 64)
#         self.maxpool1 = nn.MaxPool2d(2, 2)
#         #1
#         self.conv2 = conv_bn_relu(64, 128)
#         self.maxpool2 = nn.MaxPool2d(2, 2)
#         #2
#         self.conv3 = conv_bn_relu(128, 256)
#         self.conv4 = conv_bn_relu(256, 256)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
#         #3
#         self.conv5 = conv_bn_relu(256, 512)
#         self.conv6 = conv_bn_relu(512, 512)
#         self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
#         self.conv7 = conv_bn_relu(512, 512)
#         self.maxpool5 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
#
#         self.rnn = nn.Sequential(
#             BidirectionalLSTM(1024, number_hidden, number_hidden),
#             BidirectionalLSTM(number_hidden, number_hidden, number_class)
#         )
#
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.maxpool1(out)  # 16x64
#         out = self.conv2(out)
#         out = self.maxpool2(out)  # 8x32
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.maxpool3(out)  # 4x32
#         out = self.conv5(out)
#         out = self.conv6(out)
#         out = self.maxpool4(out) #4x16
#         out = self.conv7(out)
#         out = self.maxpool5(out) #2x16
#         b, c, h, w = out.size()
#         out = out.view(b, -1, w)
#         out = out.permute(2, 0, 1)
#         out = self.rnn(out)
#         out = F.log_softmax(out, dim=2)
#
#         return out
#
#     def backward_hook(self, module, grad_input, grad_output):
#         for g in grad_input:
#             g[g != g] = 0   # replace all nan/inf in gradients to zero


# class CRNN_Efficientnet(nn.Module):
#     def __init__(self, n_channels, number_class, number_hidden):
#         super(CRNN_Efficientnet, self).__init__()
#         self.backbone = EfficientNet.from_name("efficientnet-b2")
#         self.conv1 = conv_bn_relu(48, 512)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
#         self.rnn = nn.Sequential(
#             BidirectionalLSTM(1024, number_hidden, number_hidden),
#             BidirectionalLSTM(number_hidden, number_hidden, number_class)
#         )
#
#     def forward(self, x):
#         out = self.backbone.get_feature_b2(x)
#         out = self.conv1(out)
#         # out = self.conv2(out)
#         out = self.maxpool1(out)
#         b, c, h, w = out.size()
#         out = out.view(b, -1, w)
#         out = out.permute(2, 0, 1)
#         out = self.rnn(out)
#         out = F.log_softmax(out, dim=2)
#
#         return out
#
#
#         # test = torch.randn(1, 1, 32, 128)
#         # # model = EfficientNet.from_name("efficientnet-b2")
#         # # out = model.get_feature_b2(test)
#         # model = CRNN_Efficientnet(1, 36, 256)
#         # out = model(test)
#         # print(out.shape)
