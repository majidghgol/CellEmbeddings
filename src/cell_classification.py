import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from torch.autograd import Variable


class MyLossFunc(torch.nn.Module):
    def __init__(self, weight_matrix):
        super(MyLossFunc, self).__init__()
        self.wm = weight_matrix
        
    def forward(self, log_probs, targets):
        #return -torch.sum(torch.mul(torch.mm(targets, self.wm), log_probs))
        coeff = torch.mm(targets, self.wm)
        positive = torch.mul(targets, log_probs)
        negative = 100.0*torch.mul((1.0-targets), torch.exp(log_probs))
        #negative = -torch.mul((1.0-targets), torch.log(1.0-torch.exp(log_probs)+1e-10))
        #return -torch.mean(torch.sum(torch.mul(coeff,positive-negative), dim=1))
        #print(torch.log(1.0-torch.exp(log_probs)))
        #print(torch.sum(torch.isnan(negative)))
        #print(-torch.sum(torch.mul(coeff,positive-negative)))
        return -torch.sum(torch.mul(coeff,positive-negative))


class TableCellModel_BD(nn.Module):
    def __init__(self, dim, num_tags, cuda=False):
        super(TableCellModel_BD, self).__init__()
        self.usecuda = cuda
        self.dim1 = dim
        self.reduce_dim = nn.Linear(dim , 100)
        dim = 100
        self.dim = dim
        self.row_lstm = nn.LSTM(input_size=dim,
                                hidden_size=dim, batch_first=True, bidirectional=True)
        self.col_lstm = nn.LSTM(input_size=dim,
                                hidden_size=dim, batch_first=True, bidirectional=True)
        self.row_hidden = self.init_hidden(1)
        self.col_hidden = self.init_hidden(1)
        self.hidden2tag = nn.Linear(4 * dim, num_tags)
        self.num_tags = num_tags



    def init_hidden(self, k):
        if self.usecuda:
            return (torch.zeros(2, k, self.dim).cuda(),
                    torch.zeros(2, k, self.dim).cuda())
        else:
            return (torch.zeros(2, k, self.dim),
                    torch.zeros(2, k, self.dim))

    def forward(self, features):
        #n,m, d = features.shape
        #features = self.reduce_dim(features.view(-1, d)).view(n,m,self.dim)
        self.row_hidden = self.init_hidden(features.shape[0])
        lstm_out, self.row_hidden = self.row_lstm(
            features, self.row_hidden)
        row_tensor = lstm_out

        self.col_hidden = self.init_hidden(features.shape[1])
        lstm_out, self.col_hidden = self.col_lstm(
            features.permute(1, 0, 2), self.col_hidden)
        col_tensor = lstm_out.permute(1, 0, 2)

        #         print(row_tensor.shape)
        #         print(col_tensor.shape)

        table_tensor = torch.cat([row_tensor, col_tensor], dim=2)
        #         print(table_tensor.shape)
        tag_space = self.hidden2tag(table_tensor)
        log_probs = F.log_softmax(tag_space, dim=2)
        return log_probs


class TableCellModel(nn.Module):
    def __init__(self, dim, num_tags, cuda=False):
        super(TableCellModel, self).__init__()
        self.usecuda = cuda
        
        self.reduce_dim = nn.Linear(dim , 100)
        dim = 100
        self.dim = dim
        self.row_lstm = nn.LSTM(input_size=dim,
                                hidden_size=dim, batch_first=True)
        self.col_lstm = nn.LSTM(input_size=dim,
                                hidden_size=dim, batch_first=True)
        self.row_hidden = self.init_hidden(1)
        self.col_hidden = self.init_hidden(1)
        self.hidden2tag = nn.Linear(2 * dim, num_tags)
        self.num_tags = num_tags


    def init_hidden(self, k):
        if self.usecuda:
            return (torch.zeros(1, k, self.dim).cuda(),
                    torch.zeros(1, k, self.dim).cuda())
        else:
            return (torch.zeros(1, k, self.dim),
                    torch.zeros(1, k, self.dim))

    def forward(self, features):
        
#         self.reduce_dim(features.view(-1, )
        self.row_hidden = self.init_hidden(features.shape[0])
        lstm_out, self.row_hidden = self.row_lstm(
            features, self.row_hidden)
        row_tensor = lstm_out

        self.col_hidden = self.init_hidden(features.shape[1])
        lstm_out, self.col_hidden = self.col_lstm(
            features.permute(1, 0, 2), self.col_hidden)
        col_tensor = lstm_out.permute(1, 0, 2)

        #         print(row_tensor.shape)
        #         print(col_tensor.shape)

        table_tensor = torch.cat([row_tensor, col_tensor], dim=2)
        #         print(table_tensor.shape)
        tag_space = self.hidden2tag(table_tensor)
        log_probs = F.log_softmax(tag_space, dim=2)
        return log_probs
    
def assign_array_to_tensor(t,a, n,m,d, xshift=0, yshift=0):
    for i in range(xshift,n+xshift):
        for j in range(yshift,m+yshift):
            for k in range(d):
                t[i,j,k] = a[i-xshift,j-yshift,k]

class CellLSTMClassification:
    def __init__(self, vdim, num_classes, cuda=False):
        torch.manual_seed(12345)
        np.random.seed(12345)
        self.top_pad = np.ones([vdim]) * 2
        self.bottom_pad = np.ones([vdim]) * 3
        self.left_pad = np.ones([vdim]) * 4
        self.right_pad = np.ones([vdim]) * 5

        self.vdim = vdim
        self.model = None
        self.num_classes = num_classes
        self.cuda = cuda
        #self.model_class = TableCellModel
        self.model_class = TableCellModel_BD

    def fit(self, n_epoch, X_graph, y_graph, class_weights):
        print('this is new version')
        class_weights = torch.tensor(class_weights)
        if self.cuda:
            class_weights = class_weights.cuda()
        self.model = self.model_class(self.vdim, self.num_classes, self.cuda)
        #loss_function = nn.NLLLoss(weight=class_weights)
        loss_function = MyLossFunc(torch.tensor(np.ones([7,7]), dtype=torch.float).cuda())
        
        mlb = MultiLabelBinarizer()

        if self.cuda:
            self.model = self.model.cuda()
            loss_function = loss_function.cuda()

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for e in range(n_epoch):
            losses = 0
            for i, (x, y) in enumerate(zip(X_graph, y_graph)):
                n, m, d = x.shape
                feats = np.zeros([n + 2, m + 2, d])
                feats[1:-1, 1:-1] = x
                feats[0] = self.top_pad
                feats[-1] = self.right_pad
                feats[:, 0] = self.left_pad
                feats[:, -1] = self.right_pad
                x = torch.tensor(feats, dtype=torch.float)
                
                y = y.reshape(-1,)
                y = mlb.fit_transform(y)
                y = torch.tensor(y, dtype=torch.float)
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()

                log_probs = self.model.forward(x)
                loss = loss_function(log_probs[1:-1, 1:-1, :].contiguous().view([n * m, self.num_classes]), y)
                losses += loss.item()
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                
                del x 
                del y
                torch.cuda.empty_cache()
                print(torch.cuda.max_memory_allocated())
        return self.model
    
    def fit_iterative(self, n_epoch, X_graph, y_graph, weight_matrix, loss_method='my_loss', opt_method='adam'):
        print('this is new version')
        weight_matrix = torch.tensor(weight_matrix, dtype=torch.float)
        if self.cuda:
            weight_matrix = weight_matrix.cuda()
        
        self.model = self.model_class(self.vdim, self.num_classes, self.cuda)
        if loss_method == 'nll':
            class_weights = torch.diag(weight_matrix)
            loss_function = nn.NLLLoss(weight=class_weights, reduction='sum')
        else:
            loss_function = MyLossFunc(weight_matrix)
        
        mlb = MultiLabelBinarizer(classes=range(self.num_classes))

        if self.cuda:
            self.model = self.model.cuda()
            loss_function = loss_function.cuda()
        if opt_method == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=0.0007)

        for e in range(n_epoch):
            losses = 0
            for i, (x, y) in enumerate(zip(X_graph, y_graph)):
                n, m, d = x.shape
                feats = np.zeros([n + 2, m + 2, d])
                feats[1:-1, 1:-1] = x
                feats[0] = self.top_pad
                feats[-1] = self.right_pad
                feats[:, 0] = self.left_pad
                feats[:, -1] = self.right_pad
                x = torch.tensor(feats, dtype=torch.float)
                
#                 print(n,m,d)
                
                if loss_method == 'my_loss':
                    y = y.reshape(-1,1)
                    y = mlb.fit_transform(y)
                    y = torch.tensor(y, dtype=torch.float)
                else:
                    y = y.reshape(-1,)
                    y = torch.tensor(y, dtype=torch.long)
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                
                log_probs = self.model.forward(x)
                loss = loss_function(log_probs[1:-1, 1:-1, :].contiguous().view([n * m, self.num_classes]), y)
                losses += loss.item()
                #print(str(i), 'loss: ', loss.item())
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                
                del x 
                del y
                del log_probs
                del loss
                torch.cuda.empty_cache()
                
            yield self.model

    def fit_iterative2(self, n_epoch, X_graph, y_graph, weight_matrix, loss_method='my_loss', opt_method='adam'):
        print('this is new version2')
        
        weight_matrix = torch.tensor(weight_matrix, dtype=torch.float)
        if self.cuda:
            weight_matrix = weight_matrix.cuda()
        
        self.model = self.model_class(self.vdim, self.num_classes, self.cuda)
        
        if loss_method == 'nll':
            class_weights = torch.diag(weight_matrix)
            loss_function = nn.NLLLoss(weight=class_weights, reduction='sum')
        else:
            loss_function = MyLossFunc(weight_matrix)
        
        mlb = MultiLabelBinarizer(classes=range(self.num_classes))

        if self.cuda:
            self.model = self.model.cuda()
            loss_function = loss_function.cuda()
        if opt_method == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=0.0007)

        max_n = max([x.shape[0] for x in X_graph])
        max_m = max([x.shape[1] for x in X_graph])
        d = X_graph[0].shape[2]
        n_classes = max([np.max(x) for x in y_graph]).astype(int) + 1
        
        print(max_n, max_m, d, n_classes)
        
        feats = np.zeros([max_n + 2, max_m + 2, d], dtype=float)
        feats[0] = self.top_pad
        feats[:, 0] = self.left_pad
        Tx = torch.tensor(feats, dtype=torch.float)
        
        ys = np.zeros([max_n * max_m, n_classes], dtype=int)
        Ty = torch.tensor(ys, dtype=torch.long)
        
        if self.cuda:
            Tx = Variable(Tx.cuda(), requires_grad=False)
            Ty = Variable(Ty.cuda(), requires_grad=False)
        else:
            Tx = Variable(Tx.cpu(), requires_grad=False)
            Ty = Variable(Ty.cpu(), requires_grad=False)
        for e in range(n_epoch):
            losses = 0
            for i, (x, y) in enumerate(zip(X_graph, y_graph)):
                n, m, d = x.shape
                
                bottom_pad = np.expand_dims(np.vstack([self.bottom_pad.reshape(1,-1)]*m), 0)
                right_pad = np.expand_dims(np.vstack([self.right_pad.reshape(1,-1)]*n), 1)
                assign_array_to_tensor(Tx, x, n,m,d,1,1)
                assign_array_to_tensor(Tx, bottom_pad, 1,m,d,n,1)
                assign_array_to_tensor(Tx, right_pad, n,1,d,1,m)
#                 Tx[1:n+1, 1:m+1, :] = x
#                 Tx[n+1] = self.bottom_pad
#                 Tx[:, -1] = self.right_pad
                
                log_probs = self.model.forward(Tx[:n+2,:m+2,:])
                
                if loss_method == 'my_loss':
                    y = y.reshape(-1,1)
                    y = mlb.fit_transform(y)
                    #y = torch.tensor(y, dtype=torch.float)
#                     Ty[:,:] = y
                    assign_array_to_tensor(Ty, y, n,m,y.shape[2])
                    loss = loss_function(log_probs[1:-1, 1:-1, :].contiguous().view([n * m, self.num_classes]), Ty[:n,:m,:])
                else:
                    y = y.reshape(-1,)
                    #y = torch.tensor(y, dtype=torch.long)
                    Ty[:,0] = y
                    loss = loss_function(log_probs[1:-1, 1:-1, :].contiguous().view([n * m, self.num_classes]), Ty[:,0])
                
                losses += loss.item()
                #print(str(i), 'loss: ', loss.item())
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(torch.cuda.max_memory_allocated() / 1e6)
#                 torch.cuda.empty_cache()
                
            yield self.model

    def predict(self, X_graph):
        pred = []
        all_log_probs = []
        for i, x in enumerate(X_graph):
            n, m, d = x.shape
            feats = np.zeros([n + 2, m + 2, d])
            feats[1:-1, 1:-1] = x
            feats[0] = self.top_pad
            feats[-1] = self.right_pad
            feats[:, 0] = self.left_pad
            feats[:, -1] = self.right_pad
            x = torch.tensor(feats, dtype=torch.float)
            if self.cuda:
                x = x.cuda()

            log_probs = self.model.forward(x)
            log_probs = log_probs.detach().cpu().numpy()[1:-1, 1:-1, :]
            all_log_probs += log_probs.reshape(n*m, -1).tolist()
            p = np.argmax(log_probs, axis=2)
            pred.append(p)
            
            del x 
            torch.cuda.empty_cache()
        return pred, np.array(all_log_probs)


