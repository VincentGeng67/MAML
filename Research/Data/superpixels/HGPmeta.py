import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import tensorflow as tf

from    HGPLearner import HGPLearner
from    copy import deepcopy



class HGPMeta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self,update_lr,meta_lr,way,update_step,update_step_test):
        """
        :param args:
        """
        super(HGPMeta, self).__init__()
        print('parameters',update_lr,meta_lr)
        self.update_lr =update_lr
        self.meta_lr = meta_lr
#         self.n_way = args.n_way
#         self.k_spt = args.k_spt
#         self.k_qry = args.k_qry
#         self.task_num = args.task_num
        self.update_step = update_step
        self.update_step_test = update_step_test
        
#         self.update_step_test = args.update_step_test


        self.net = HGPLearner(way)

        
        





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
        metagrad=[]
         
        for i in range(task_num):

            score,logits,loss,correct= self.net(x_spt[i],'train', vars=None, bn_training=True,init=True)
            del score
            del logits
            del correct
       


            # grad = torch.autograd.grad(loss, self.net.parameters(),retain_graph=True,create_graph=True)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
#             enlist=[0,1,3,5,6,7,9,11,12,13]
#             print('checkeffectbefore',grad[0])  
#             for l1 in  enlist:
#                 print('second order',l1)
#                 length=0
#                 more=0
#                 if (l1 ==12):
#                     length=len(grad[l1]) 
#                     more=2
#                 elif (len(grad[l1])>1):
#                     length=len(grad[l1])
#                     more=0
#                 else :
#                     length=len(grad[l1][0])
#                     more=1
#                 for l2 in range(length):
                    
#                     if (more==1) :
# #                         print('element',grad[l1][0][l2])
#                         s_grads = torch.autograd.grad(grad[l1][0][l2], self.net.parameters(),allow_unused=True,retain_graph=True)
#                         grad[l1][0][l2]=s_grads[l1][0][l2]
#                         del s_grads
#                     elif(more==0):
# #                         print('element',grad[l1][l2])
#                         s_grads = torch.autograd.grad(grad[l1][l2], self.net.parameters(),allow_unused=True,retain_graph=True)
#                         grad[l1][l2]=s_grads[l1][l2]
#                         del s_grads
#                     else:
#                         for l3 in range(len(grad[l1][l2])): 
# #                             print('element',grad[l1][l2][l3])
#                             s_grads = torch.autograd.grad(grad[l1][l2][l3], self.net.parameters(),allow_unused=True,retain_graph=True)
#                             grad[l1][l2][l3]=s_grads[l1][l2][l3]
#                             del s_grads
#             print('checkeffect',grad[0])             
            sgrad = list(map(lambda p: 1 - self.update_lr * p[0], zip(grad)))
            
            # f = torch.autograd.grad(loss, self.net.parameters())
            # del f
            del grad

            


            
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]

                score_q,logits_q,loss_q,correct= self.net(x_qry[i],'test', self.net.parameters(), bn_training=True,init=True)
                del score_q
                del logits_q
                losses_q[0] += float(loss_q)

                corrects[0] = corrects[0] + float(correct)
# this is the loss and accuracy after the first update

            with torch.no_grad():
                # [setsz, nway]
                
                score_q,logits_q,loss_q,correct= self.net(x_qry[i],'test', fast_weights, bn_training=True)
#                 loss_q = criterion(score_q, y_qry[i])
#                 loss_q=mloss_q
                del score_q
                del logits_q
                losses_q[1] += float(loss_q)
                # [setsz]
                corrects[1] = corrects[1] + float(correct)
#                 print('correct',correct)




            
            

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
#                 logits = self.net(x_spt[i], fast_weights, bn_training=True)
#                 loss = F.cross_entropy(logits, y_spt[i])
                score,logits,loss,correct= self.net(x_spt[i],'train', fast_weights, bn_training=True)

                del score
                del logits
                del correct
                
                # grad = torch.autograd.grad(loss,self.net.modelbn,retain_graph=True,create_graph=True)
                grad = torch.autograd.grad(loss,self.net.modelbn)
#                 print('checkgrad1',grad[0])
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
#                 enlist=[0,1,3,5,6,7,9,11,12,13]
#                 for l1 in  enlist:
#                     print('second order',l1)
#                     length=0
#                     more=0
#                     if (l1 ==12):
#                         length=len(grad[l1]) 
#                         more=2
#                     elif (len(grad[l1])>1):
#                         length=len(grad[l1])
#                         more=0
#                     else :
#                         length=len(grad[l1][0])
#                         more=1
#                     for l2 in range(length):
                    
#                         if (more==1) :
# #                         print('element',grad[l1][0][l2])
#                             s_grads = torch.autograd.grad(grad[l1][0][l2], self.net.modelbn,allow_unused=True,retain_graph=True)
#                             grad[l1][0][l2]=s_grads[l1][0][l2]
#                             del s_grads
#                         elif(more==0):
# #                         print('element',grad[l1][l2])
#                             s_grads = torch.autograd.grad(grad[l1][l2], self.net.modelbn,allow_unused=True,retain_graph=True)
#                             grad[l1][l2]=s_grads[l1][l2]
#                             del s_grads
#                         else:
#                             for l3 in range(len(grad[l1][l2])): 
# #                             print('element',grad[l1][l2][l3])
#                                 s_grads = torch.autograd.grad(grad[l1][l2][l3], self.net.modelbn,allow_unused=True,retain_graph=True)
#                                 grad[l1][l2][l3]=s_grads[l1][l2][l3]
#                                 del s_grads
                grad = list(map(lambda p: 1 - self.update_lr * p[0], zip(grad)))
                sgrad= list(map(lambda p: p[0] * p[1], zip(grad,sgrad)))
                del grad
                # f = torch.autograd.grad(loss, self.net.parameters())
                # del f

                score_q,logits_q,loss_q,correct= self.net(x_qry[i],'test', fast_weights, bn_training=True)
                losses_q[k + 1] += float(loss_q)
    
                del score_q
                del logits_q
            
                if(k==self.update_step-1):
                    fgrad=torch.autograd.grad(loss_q, self.net.modelbn)
                    print('final ratio grad',sgrad[0])
                    print('final ratio grad',sgrad[13])
                    sgrad= list(map(lambda p: p[0] * p[1], zip(fgrad,sgrad)))
                    del fgrad
                    del fast_weights
                    metagrad.append(sgrad)
                    print('final step grad',sgrad[0])
                    print('final step grad',sgrad[13])
                with torch.no_grad():

                    corrects[k + 1] = corrects[k + 1] + float(correct)
            


        up_grad=list(map(lambda p: p[0]/ task_num, zip(metagrad[0])))
        
        for tas in range(1,len(metagrad)):
            up_grad= list(map(lambda p: p[0] + p[1]/ task_num, zip(up_grad,metagrad[tas])))


        del metagrad
        print('beforepara',self.net.varstest[0])
        print('beforepara',self.net.varstest[13])
        newpara=list(map(lambda p: p[0]-self.meta_lr*p[1], zip(self.net.varstest,up_grad)))
        del up_grad
        print('newpara',newpara[0])
        print('newpara',newpara[13])
        self.net.model.setpara(newpara,False)
        self.net.varstest=self.net.model.weight_bias()
        print('afterpara',self.net.parameters()[0])
        print('afterpara',self.net.parameters()[13])

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
#         net = deepcopy(self.net)
        net = HGPLearner(self.net.way)
        net.model=self.net.model
        net.varstest=self.net.varstest
        net.model.conv2.cached_result=self.net.model.conv2.cached_result
        net.model.conv2.cached_num_edges=self.net.model.conv2.cached_num_edges
        net.model.conv3.cached_result=self.net.model.conv3.cached_result
        net.model.conv3.cached_num_edges=self.net.model.conv3.cached_num_edges
        net.model.pool1.calc_information_score.cached_result=self.net.model.pool1.calc_information_score.cached_result
        net.model.pool1.calc_information_score.cached_num_edges=self.net.model.pool1.calc_information_score.cached_num_edges
        net.model.pool2.calc_information_score.cached_result=self.net.model.pool2.calc_information_score.cached_result
        net.model.pool2.calc_information_score.cached_num_edges=self.net.model.pool2.calc_information_score.cached_num_edges
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
#             print('fw ft',k,fast_weights[0])
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