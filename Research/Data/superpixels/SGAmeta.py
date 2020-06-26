import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import tensorflow as tf

from    SAGLearner import SAGLearner
from    copy import deepcopy



class SGAMeta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self,update_lr,meta_lr,way):
        """
        :param args:
        """
        super(SGAMeta, self).__init__()
        print('parameters',update_lr,meta_lr)
        self.update_lr =update_lr
        self.meta_lr = meta_lr
#         self.n_way = args.n_way
#         self.k_spt = args.k_spt
#         self.k_qry = args.k_qry
#         self.task_num = args.task_num
        self.update_step = 3
        self.update_step_test=3
        
#         self.update_step_test = args.update_step_test


        self.net = SAGLearner(way)

        
        self.meta_optim = optim.Adam(self.net.parameters(), lr=meta_lr, weight_decay=0.0001)





    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt,x_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
#         task_num, setsz, c_, h, w = x_spt.size()
        task_num=len(x_spt)
#         print('size',querysz)
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):

            score,logits,loss,correct= self.net(x_spt[i],'train', vars=None, bn_training=True,init=True)
        
       

            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))


            
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]

                score_q,logits_q,loss_q,correct= self.net(x_qry[i],'test', self.net.parameters(), bn_training=True,init=True)

                losses_q[0] += loss_q

                corrects[0] = corrects[0] + correct
# this is the loss and accuracy after the first update

            with torch.no_grad():
                # [setsz, nway]
                
                score_q,logits_q,loss_q,correct= self.net(x_qry[i],'test', fast_weights, bn_training=True)
#                 loss_q = criterion(score_q, y_qry[i])
#                 loss_q=mloss_q

                losses_q[1] += loss_q
                # [setsz]
                corrects[1] = corrects[1] + correct
#                 print('correct',correct)




            
            
            
            
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
#                 logits = self.net(x_spt[i], fast_weights, bn_training=True)
#                 loss = F.cross_entropy(logits, y_spt[i])
                score,logits,loss,correct= self.net(x_spt[i],'train', fast_weights, bn_training=True)
                
    
                
                grad = torch.autograd.grad(loss,self.net.modelbn)
                print('checkgrad1',grad[0])
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                print('fwit',k,fast_weights[0])
#                 print('sorrypred1',logits)
#                 print('sorrylabel2',y_spt[i])
                score_q,logits_q,loss_q,correct= self.net(x_qry[i],'test', fast_weights, bn_training=True)

#                 loss_q = criterion(score_q, y_qry[i])
                losses_q[k + 1] += loss_q


                with torch.no_grad():

                    corrects[k + 1] = corrects[k + 1] + correct
            





        loss_q = losses_q[-1] / task_num
        print('beforepara',self.net.parameters()[0])
        self.meta_optim.zero_grad()
        loss_q.backward()
        print('grad',self.net.parameters()[0].grad)
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()
        print('afterpara',self.net.parameters()[0])

        accs = np.array(corrects) / (task_num)
        
        return accs,loss_q


    def finetunning(self, x_spt, x_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
#         assert len(x_spt.shape) == 4
#         print('hello')
#         querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        print('newpara',net.parameters()[0])
        # 1. run the i-th task and compute loss for k=0
        score,logits,loss,correct= net(x_spt,'train',init=True)
        grad = torch.autograd.grad(loss, net.parameters())
#         grad = torch.autograd.grad(loss, net.parameters(),retain_graph=True,allow_unused=True)
#         print('ftcheckgrad',grad[0])
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))


        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
#             logits_q = net(x_qry, net.parameters(), bn_training=True)
            score_q,logits_q,loss_q,correct= net(x_qry,'test', net.parameters(), bn_training=True,init=True)

            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
#             logits_q = net(x_qry, fast_weights, bn_training=True)
            score_q,logits_q,loss_q,correct= net(x_qry,'test', fast_weights, bn_training=True)

            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            score,logits,loss,correct= net(x_spt,'train', fast_weights, bn_training=True)

#             logits = net(x_spt, fast_weights, bn_training=True)
#             loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, net.modelbn)
#             print('gradit ft',k,grad[0])

#             grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            print('fw ft',k,fast_weights[0])
            score_q,logits_q,loss_q,correct= net(x_qry,'test', fast_weights, bn_training=True)

#             loss_q = criterion(score_q, y_qry)
#             loss_q=mloss_q
            
#             logits_q = net(x_qry, fast_weights, bn_training=True)
#             # loss_q will be overwritten and just keep the loss_q on last update step.
#             loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():

                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) 

        return accs




def main():
    pass


if __name__ == '__main__':
    main()