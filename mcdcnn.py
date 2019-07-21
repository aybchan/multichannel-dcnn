import torch
import torch.nn as nn
import torch.nn.functional as F

class Channel(nn.Module):
    def __init__(self,n_channels,time_steps,strides,filter_maps):
        super(Channel, self).__init__()
        self.time_steps = time_steps
        self.strides = strides
        self.filter_maps = filter_maps
        self.conv1 = nn.Conv1d(1,self.filter_maps[0],self.strides[0])
        self.bn1 = nn.BatchNorm1d(self.filter_maps[0])
        self.conv2 = nn.Conv1d(self.filter_maps[1],
                               self.filter_maps[2],
                               self.strides[1])
        self.bn2 = nn.BatchNorm1d(self.filter_maps[2])

        self.channel_out = int(((self.time_steps - (self.strides[0]-1))/2
                            -(self.strides[1]-1))/2)

    def forward(self, x):
        x = x.view(-1,1,self.time_steps)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x,2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x,2)

        return x.view(-1,self.channel_out)

class MCDCNN(nn.Module):
    def __init__(self,n_channels,time_steps,n_classes,strides, filter_maps):
        super(MCDCNN, self).__init__()
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.n_classes = n_classes
        self.channels = nn.ModuleList()
        self.strides = strides
        self.filter_maps = filter_maps
        self.channel_out = int(((self.time_steps - (self.strides[0]-1))/2
                            -(self.strides[1]-1))/2)

        for channel in range(self.n_channels):
            self.channels.extend([Channel(n_channels,time_steps,
                                          self.strides,self.filter_maps)])

        self.fc1 = nn.Linear( self.n_channels*self.channel_out, 256)
        self.fc2 = nn.Linear( 256, 128)
        self.fc3 = nn.Linear( 128,  32)
        self.fc4 = nn.Linear(  32, self.n_classes)

    def forward(self, x):
        outs = []
        for i,channel in enumerate(self.channels):
            out = channel(x[:,:,i,:])
            outs.append(out)

        n = out.size()[-1] * len(outs)
        # concatenate channel outs
        x = torch.cat(outs,dim=0)
        x = x.view(-1,n)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return F.log_softmax(x)
