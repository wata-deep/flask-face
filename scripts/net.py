import torch
import torch.nn as nn
import torch.nn.functional as F

# Network

class Reshape(nn.Module):
    def __init__(self, h, w):
        super(Reshape, self).__init__()
        self.h = h
        self.w = w

    def forward(self, x):
        return x.view(x.size()[0], -1, self.h, self.w)

class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel=None, norm=nn.BatchNorm2d, down=False):
        super(ResBlock, self).__init__()
        stride = 2 if down else 1

        if out_channel or down:
            self.Conv3 = nn.Conv2d(in_channel, out_channel, 1, stride, bias=False)
            self.No3 = norm(out_channel)
        else:
            out_channel = in_channel

        self.Conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1)
        self.No1   = norm(out_channel)
        self.Conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.No2   = norm(out_channel)

        self.Act   = nn.ReLU()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.Act(self.No1(self.Conv1(x)))
        y = self.No2(self.Conv2(y))

        if x.size() != y.size():
            x = self.No3(self.Conv3(x))

        return self.Act(x + y)

class ConditionMarge(nn.Module):
    def __init__(self, in_channel, label_size, outvec_size):
        super(ConditionMarge, self).__init__()
        self.Fc = nn.Linear(label_size, outvec_size)
        self.Marge = nn.Conv2d(in_channel + 1, in_channel, 1, 1, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, label):
        b, c, h, w = x.size()
        y_label = self.Fc(label).view(b, 1, h, w)
        y = torch.cat((x, y_label), dim=1)
        y = self.Marge(y)

        return y

class CResBlock(nn.Module): # Condition ResBlock
    def __init__(self, in_channel, out_channel=None, label_size=None, fmap_size=None, down=False, normalize=nn.BatchNorm2d):
        assert label_size and fmap_size, "label_size or fmap_size is \"None\""
        super(CResBlock, self).__init__()
        stride = 2 if down else 1

        outvec_size = fmap_size[0] * fmap_size[1]
        if out_channel or down:
            self.Marge3 = ConditionMarge(in_channel, label_size, outvec_size)
            self.Conv3  = nn.Conv2d(in_channel, out_channel, 1, stride, bias=False)
            self.No3    = normalize(out_channel)
        else:
            out_channel = in_channel

        self.Marge1 = ConditionMarge(in_channel, label_size, outvec_size)
        self.Conv1  = nn.Conv2d(in_channel, out_channel, 3, stride, 1)
        self.No1    = normalize(out_channel)
        self.Marge2 = ConditionMarge(out_channel, label_size, outvec_size // (stride * stride))
        self.Conv2  = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.No2    = normalize(out_channel)

        self.Act   = nn.ReLU()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label):
        y = self.Act(self.No1(self.Conv1(self.Marge1(x, label))))
        y = self.No2(self.Conv2(self.Marge2(y, label)))

        if x.size() != y.size():
            x = self.No3(self.Conv3(self.Marge3(x, label)))

        return self.Act(x + y)

class Gen(nn.Module):
    def __init__(self, in_channel, label_channel):
        super(Gen, self).__init__()
        self.Fc = nn.Linear(in_channel, 4 * 4 * 1024)
        self.Reshape = Reshape(4, 4)
        self.CRB1 = CResBlock(1024, out_channel=512, label_size=label_channel, fmap_size=(4, 4), normalize=nn.InstanceNorm2d)
        self.Bilinear1 = nn.Upsample(scale_factor=2., mode="bilinear", align_corners=True)
        self.CRB2 = CResBlock(512, out_channel=256, label_size=label_channel, fmap_size=(8, 8), normalize=nn.InstanceNorm2d)
        self.Bilinear2 = nn.Upsample(scale_factor=2., mode="bilinear", align_corners=True)
        self.CRB3 = CResBlock(256, out_channel=128, label_size=label_channel, fmap_size=(16, 16), normalize=nn.InstanceNorm2d)
        self.Bilinear3 = nn.Upsample(scale_factor=2., mode="bilinear", align_corners=True)
        self.CRB4 = CResBlock(128, out_channel=64, label_size=label_channel, fmap_size=(32, 32), normalize=nn.InstanceNorm2d)
        self.Bilinear4 = nn.Upsample(scale_factor=2., mode="bilinear", align_corners=True)
        self.CRB5 = CResBlock(64, out_channel=32, label_size=label_channel, fmap_size=(64, 64), normalize=nn.InstanceNorm2d)
        self.Bilinear5 = nn.Upsample(scale_factor=2., mode="bilinear", align_corners=True)
        self.Conv = nn.Conv2d(32, 3, 3, 1, 1)
        self.Tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label):
        y = self.Reshape(self.Fc(x))
        y = self.Bilinear1(self.CRB1(y, label))
        y = self.Bilinear2(self.CRB2(y, label))
        y = self.Bilinear3(self.CRB3(y, label))
        y = self.Bilinear4(self.CRB4(y, label))
        y = self.Bilinear5(self.CRB5(y, label))
        y = self.Tanh(self.Conv(y))

        return y


class Enc(nn.Module):
    def __init__(self, out_channel):
        super(Enc, self).__init__()
        self.Module = nn.Sequential()
        self.Module.add_module("Conv", nn.Conv2d(3, 32, 7, 2, 3, bias=False)) # 64x64
        self.Module.add_module("Act", nn.ReLU())
        self.Module.add_module("Res1", ResBlock(32, 64, down=True)) # 32x32
        self.Module.add_module("Res2", ResBlock(64, 128, down=True)) # 16x16
        self.Module.add_module("Res3", ResBlock(128, 256, down=True)) # 8x8
        self.Module.add_module("Res4", ResBlock(256, 512, down=True)) # 4x4
        self.Module.add_module("Flatten", Flatten())
        self.Module.add_module("Fc", nn.Linear(4 * 4 * 512, out_channel, bias=False))
        self.Module.add_module("Out", nn.Sigmoid())

        for m in self.Module.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.Module(x)

class Net_ele(nn.Module):
    def __init__(self):
        super(Net_ele, self).__init__()
        self.In    = nn.InstanceNorm2d(3)
        self.Conv1 = nn.Conv2d(3, 64, 3, 1, 1) # 128
        self.Bn1   = nn.BatchNorm2d(64)
        self.Conv2 = nn.Conv2d(64, 64, 3, 1, 1) # 128
        self.Bn2   = nn.BatchNorm2d(64)
        self.Conv3 = nn.Conv2d(64, 128, 2, 2) # 64
        self.Bn3   = nn.BatchNorm2d(128)
        self.Conv4 = nn.Conv2d(128, 128, 3, 1, 1) #64
        self.Bn4   = nn.BatchNorm2d(128)
        self.Conv5 = nn.Conv2d(128, 256, 2, 2) #32
        self.Bn5   = nn.BatchNorm2d(256)
        self.Conv6 = nn.Conv2d(256, 256, 3, 1, 1) #32
        self.Bn6   = nn.BatchNorm2d(256)
        self.Conv7 = nn.Conv2d(256, 512, 2, 2) # 16
        self.Bn7   = nn.BatchNorm2d(512)
        self.Conv8 = nn.Conv2d(512, 512, 3, 1, 1) # 16
        self.Bn8   = nn.BatchNorm2d(512)
        self.Conv9 = nn.Conv2d(512, 1024, 1, 1) # 16
        self.Bn9   = nn.BatchNorm2d(1024)

        self.Fc1    = nn.Linear(1024, 2048)
        self.Age    = nn.Linear(2048, 1)
        self.Gender = nn.Linear(2048, 2)
        self.Race   = nn.Linear(2048, 4)
        self.Smile  = nn.Linear(2048, 1)
        self.Points = nn.Linear(2048, 10)

    def forward(self, x):
        y = self.In(x)
        y = F.relu(self.Bn1(self.Conv1(y)))
        y = F.relu(self.Bn2(self.Conv2(y)))
        y = F.relu(self.Bn3(self.Conv3(y)))
        y = F.relu(self.Bn4(self.Conv4(y)))
        y = F.relu(self.Bn5(self.Conv5(y)))
        y = F.relu(self.Bn6(self.Conv6(y)))
        y = F.relu(self.Bn7(self.Conv7(y)))
        y = F.relu(self.Bn8(self.Conv8(y)))
        y = F.relu(self.Bn9(self.Conv9(y)))

        y = F.avg_pool2d(y, y.size()[2:]).view(y.size()[0], -1)

        y = F.relu(self.Fc1(y))

        age = torch.sigmoid(self.Age(y))
        gender = self.Gender(y)
        race = self.Race(y)
        smile = torch.sigmoid(self.Smile(y))
        points = torch.sigmoid(self.Points(y))

        return age, gender, race, smile, points
