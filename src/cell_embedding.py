import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle

class TableCellModel_SG(nn.Module):
    def __init__(self, vdim, encdim, num_context, bn):
        super(TableCellModel_SG, self).__init__()
        self.vdim = vdim
        self.encoder = nn.Linear(vdim * num_context, encdim)
        self.decoder = nn.Linear(encdim, vdim)

        self.dropout = nn.Dropout(p=0.3)

        self.bn = nn.BatchNorm1d(vdim * num_context)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.nl2l = nn.Linear(2 * encdim, encdim)
        self.usebn = bn
        
        self.train_encoder = nn.Sequential(nn.Dropout(p=0.3),
                                           nn.Linear(vdim, encdim),
                                           nn.Dropout(p=0.1))
        
        self.test_encoder = nn.Sequential(nn.ReLU(),
                                           nn.Linear(vdim, encdim))
        
        self.decoder = nn.Sequential(nn.Linear(encdim, vdim),
                                     nn.Linear(vdim, vdim * num_context))

    def init_hidden(self):
        return (torch.zeros(1, 1, self.vdim),
                torch.zeros(1, 1, self.vdim))

    def cell_forward(self, words):
        word_vecs = self.W[words]
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.cell_lstm(
            word_vecs.view(len(words), 1, -1), self.hidden)
        return lstm_out[-1].view(1, self.vdim)

    def forward(self, target_vecs, doc_ids, train=True):
        """
        target_vecs: (batch_size, 1, vdim)
        """
        if train:
            ench = self.train_encoder(target_vecs)
            dech = self.decoder(ench)
        else:
            ench = self.test_encoder(target_vecs)
            dech = self.decoder(ench)
       
        return dech.squeeze(), ench.squeeze()
    
class TableCellModel_CBOW(nn.Module):
    def __init__(self, vdim, encdim, num_context, bn):
        super(TableCellModel_CBOW, self).__init__()
        self.vdim = vdim
        self.train_encoder = nn.Sequential(nn.Dropout(p=0.3), 
                                           nn.Linear(vdim * num_context, encdim))
        
        self.test_encoder = nn.Sequential(nn.Linear(vdim * num_context, encdim))
        
        self.decoder = nn.Sequential(nn.Linear(encdim, vdim))
        

    def init_hidden(self):
        return (torch.zeros(1, 1, self.vdim),
                torch.zeros(1, 1, self.vdim))

    def cell_forward(self, words):
        word_vecs = self.W[words]
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.cell_lstm(
            word_vecs.view(len(words), 1, -1), self.hidden)
        return lstm_out[-1].view(1, self.vdim)

    def forward(self, context_vecs, doc_ids, train=True):
        """
        context_vecs: (batch_size, num_context, vdim)
        """
        h1 = context_vecs
        #h = torch.sum(h1, dim=1) # sum of context vectors
        h = h1.view(h1.shape[0], -1)  # concatenation of context vectors
        if train:
            ench = self.train_encoder(h)
            dech = self.decoder(ench)
        else:
            ench = self.test_encoder(h)
            dech = self.decoder(ench)
        return dech.squeeze(), ench.squeeze()

class TableCellModel_DM_InferSent(nn.Module):
    def __init__(self, vdim, encdim, num_context, bn):
        super(TableCellModel_DM_InferSent, self).__init__()
        self.vdim = vdim
        self.encoder = nn.Linear(vdim * num_context, encdim)
        self.decoder = nn.Linear(encdim, vdim)

        self.dropout = nn.Dropout(p=0.3)

        self.bn = nn.BatchNorm1d(vdim * num_context)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.nl2l = nn.Linear(2 * encdim, encdim)
        self.usebn = bn

    def init_hidden(self):
        return (torch.zeros(1, 1, self.vdim),
                torch.zeros(1, 1, self.vdim))

    def cell_forward(self, words):
        word_vecs = self.W[words]
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.cell_lstm(
            word_vecs.view(len(words), 1, -1), self.hidden)
        return lstm_out[-1].view(1, self.vdim)

    def forward(self, context_vecs, doc_ids, train=True):
        """
        context_vecs: (batch_size, num_context, vdim)
        """
        h1 = context_vecs
        #h = torch.sum(h1, dim=1) # sum of context vectors
        h = h1.view(h1.shape[0], -1)  # concatenation of context vectors
        #if self.usebn:
        #h = self.bn(h)
        if train:
            h = self.dropout(h)
        ench = self.encoder(h.unsqueeze(1))
        dech = self.decoder(ench)
        return dech.squeeze(), ench.squeeze()


class CellEmbedding:
    def __init__(self, vdim, encdim, num_context, bn=False, cuda=False):
        torch.manual_seed(12345)
        np.random.seed(12345)
        self.vdim = vdim
        self.encdim = encdim
        self.num_context = num_context
        self.bn = bn
        self.model = None
        self.curr_cell_ind = 0
        self.sent_enc = None
        self.s2i = None
        self.cuda = cuda

    def get_next_batch(self, cells, batch_size, num_batches, num_cells):
        for bi in range(num_batches):
            batch_count = 0
            target_texts = []
            context_texts = []
            batch_dict = dict()
            while self.curr_cell_ind < num_cells and batch_count < batch_size:
                # c = cells[self.curr_cell_ind]
                c = cells.next()
                text = c['text']
                rtext = c['rightText']
                ltext = c['leftText']
                ttext = c['topText']
                btext = c['bottomText']

                rtext1 = c['rightText1']
                ltext1 = c['leftText1']
                ttext1 = c['topText1']
                btext1 = c['bottomText1']

                context_texts.append([rtext, ltext, ttext, btext,
                                      rtext1, ltext1, ttext1, btext1])
                target_texts.append(text)
                batch_count += 1
                self.curr_cell_ind += 1
            context_vecs = np.zeros([1, batch_count, 8, 4096], dtype='float32')
            target_vecs = np.zeros([1, batch_count, 4096], dtype='float32')
            
            context_vecs[0] = [[self.sent_enc[self.s2i[xx]] if xx != '' else np.zeros([4096]) for xx in x] for x in context_texts]
            context_vecs = torch.tensor(context_vecs, dtype=torch.float32)
            if self.cuda:
                context_vecs = context_vecs.cuda()

            target_vecs[0] = [self.sent_enc[self.s2i[xx]] for xx in target_texts]
            target_vecs = torch.tensor(target_vecs, dtype=torch.float32)
            if self.cuda:
                target_vecs = target_vecs.cuda()
            return context_vecs, target_vecs

    def fit_transform(self, cells, num_cells, num_epochs, num_batches, batch_size, sent_enc, s2i):
        self.fit(cells, num_epochs, num_cells, num_batches, batch_size, sent_enc, s2i)
        return self.transform(cells)

    def fit(self, cells, num_cells, num_epochs, num_batches, batch_size, sent_enc, s2i):
        self.model = TableCellModel_DM_InferSent(self.vdim, self.encdim, self.num_context, self.bn)
        if self.cuda:
            self.model = self.model.cuda()
        cost_func = nn.MSELoss(reduction='sum')
        if self.cuda:
            cost_func = cost_func.cuda()

        optimizer = optim.Adam(params=self.model.parameters(), lr=0.001)

        self.sent_enc = sent_enc
        self.s2i = s2i

        # num_cells = len(cells)
        for e in range(num_epochs):
            cells.reset_iterator()
            e_time = time.time()
            losses = 0
            self.curr_cell_ind = 0
            progress = [20, 40, 60, 80, 100]
            tot_t = 0
            b_time = time.time()
            bi = 0
            t = time.time()
            while bi < num_batches:
                tot_t += time.time() - t
                t = time.time()
                context_vecs, target_vecs = self.get_next_batch(cells, batch_size, num_batches, num_cells)
                k = int(bi / num_batches * 100.0)
                if k in progress:
                    print(k, '% done. avg batch time: ', '%.2f' % (tot_t / (num_batches / 5)), ' avg loss: ',
                          '%.4f' % (losses / bi))
                    progress.remove(k)

                    tot_t = 0

                preds, _ = self.model.forward(context_vecs[0], None)
                loss = cost_func(preds, target_vecs[0])

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                b_time = time.time() - b_time
                # logging.debug(
                #     'epoch [{}/{}], batch [{}/{}], loss: {}, time: {}, qsize: {}'.format(e + 1, num_epochs, bi + 1,
                #                                                                          num_batches, loss.item(),
                #                                                                          b_time, batch_queue.qsize()))
                # print('epoch [{}/{}], batch [{}/{}], loss: {}, time: {}'.format(e+1, num_epochs, bi+1, num_batches, loss.item(), b_time))
                b_time = time.time()
                bi += 1
                del context_vecs
                del target_vecs
                torch.cuda.empty_cache()
            # test_clf_report = validate_accuracy(model, cells_target_processed, labels_target)
            #     print('cv on sourcee', source_cv_score, np.mean(source_cv_score))
            #     print('cv on train: ', train_cv_score, np.mean(train_cv_score))
            # print(test_clf_report)
            e_time = time.time() - e_time
            # logging.info(test_clf_report)
            # logging.info('epoch {}: avg loss: {}, time: {}'.format(e, losses / num_batches, e_time))
            print('epoch {}: avg loss: {}, time: {}'.format(e, losses / num_batches, e_time))
        return self.model
        
    def fit_iterative(self, cells, num_cells, num_epochs, num_batches, batch_size, sent_enc, s2i):
        self.model = TableCellModel_DM_InferSent(self.vdim, self.encdim, self.num_context, self.bn)
        if self.cuda:
            self.model = self.model.cuda()
        cost_func = nn.MSELoss(reduction='sum')
        if self.cuda:
            cost_func = cost_func.cuda()

        optimizer = optim.Adam(params=self.model.parameters(), lr=0.001)

        self.sent_enc = sent_enc
        self.s2i = s2i

        # num_cells = len(cells)
        for e in range(num_epochs):
            cells.reset_iterator()
            e_time = time.time()
            losses = 0
            self.curr_cell_ind = 0
            progress = [20, 40, 60, 80, 100]
            tot_t = 0
            b_time = time.time()
            bi = 0
            t = time.time()
            while bi < num_batches:
                tot_t += time.time() - t
                t = time.time()
                context_vecs, target_vecs = self.get_next_batch(cells, batch_size, num_batches, num_cells)
                k = int(bi / num_batches * 100.0)
                if k in progress:
                    print(k, '% done. avg batch time: ', '%.2f' % (tot_t / (num_batches / 5)), ' avg loss: ',
                          '%.4f' % (losses / bi))
                    progress.remove(k)

                    tot_t = 0

                preds, _ = self.model.forward(context_vecs[0], None)
                loss = cost_func(preds, target_vecs[0])

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                b_time = time.time() - b_time
                # logging.debug(
                #     'epoch [{}/{}], batch [{}/{}], loss: {}, time: {}, qsize: {}'.format(e + 1, num_epochs, bi + 1,
                #                                                                          num_batches, loss.item(),
                #                                                                          b_time, batch_queue.qsize()))
                # print('epoch [{}/{}], batch [{}/{}], loss: {}, time: {}'.format(e+1, num_epochs, bi+1, num_batches, loss.item(), b_time))
                b_time = time.time()
                bi += 1
                del context_vecs
                del target_vecs
                torch.cuda.empty_cache()
            # test_clf_report = validate_accuracy(model, cells_target_processed, labels_target)
            #     print('cv on sourcee', source_cv_score, np.mean(source_cv_score))
            #     print('cv on train: ', train_cv_score, np.mean(train_cv_score))
            # print(test_clf_report)
            e_time = time.time() - e_time
            # logging.info(test_clf_report)
            # logging.info('epoch {}: avg loss: {}, time: {}'.format(e, losses / num_batches, e_time))
            print('epoch {}: avg loss: {}, time: {}'.format(e, losses / num_batches, e_time))
            yield self.model

    def load_model(self, model_path, sent_enc, s2i):
        self.model = pickle.load(open(model_path, 'rb'))
        self.sent_enc = sent_enc
        self.s2i = s2i

    def transform(self, cells, num_cells, num_batches, batch_size):
        if self.model is None:
            raise Exception('should fit first!')
        embeddings = np.zeros([num_cells, self.encdim], dtype='float32')
        self.curr_cell_ind = 0
        with torch.no_grad():
            for i in range(num_batches):
                context_vecs, lp = self.get_next_batch(cells, batch_size, num_batches, num_cells)
                xxx, emb = self.model.forward(context_vecs[0], None, train=False)

                embeddings[i*batch_size:(i+1)*batch_size,:] = emb.detach().cpu().numpy()
                del context_vecs
                del xxx
                del lp
                torch.cuda.empty_cache()
        return embeddings