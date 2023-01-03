from torch.utils.tensorboard import SummaryWriter
import os
import time
import util
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, roc_auc_score, recall_score
import warnings
warnings.filterwarnings('ignore')

class Trainer():
    def __init__(self, args, subjectList, flatten_subjectList, subject_id, model):
        self.subjectList=subjectList
        self.subject_id=subject_id
        self.args=args
        self.model=model
        self.flatten_subjectList=flatten_subjectList
        self.writer = SummaryWriter(f'{self.args["total_path"]}/{self.flatten_subjectList[self.subject_id]}') # tensorboard, log directory
        self.num_domain=len(self.flatten_subjectList)-1
        
        self.total_domain_label=[]
        
        self.metric_dict={"loss":0, "acc":1, "bacc":2, "f1":3}

        self.set_optimizer()
        self.set_learning_rate_Scheduler()
    
    '''
    ###########################################################################################
    #  Initialize parameter
    ###########################################################################################
    '''
    def set_optimizer(self):
        if self.args['optimizer']=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer']=='AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer']=='SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=self.args['momentum'], weight_decay=self.args['weight_decay'], nesterov=self.args['nestrov'])
    
    def set_learning_rate_Scheduler(self):
        if self.args['scheduler']=="CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args['steps']//self.args['checkpoint_freq'])
            
    
    '''
    ###########################################################################################
    #  Train and Predict
    ###########################################################################################
    '''
    def training(self, train_loaders, valid_loader, test_loader):
        train_minibatches = zip(*train_loaders) # gather same number of samples from each subject (domain-balanced mini-batch)

        best_score = [100.0, 0.0, 0.0, 0.0]
        start_step=0
        self.losses=[]
        
        for step in range(start_step, self.args['steps']):
            minibatches_device = [(x.to(self.args['device']), y.to(self.args['device'])) for x,y,_ in next(train_minibatches)] # data, class_label, domain_label
            
            self.train(minibatches_device, step) # train

            if (step % self.args['checkpoint_freq'] == 0) or (step == self.args['steps'] - 1):
                valid_score = self.eval("valid", valid_loader, step) # valid 
                self.eval("test", test_loader, step) # test
                self.scheduler.step()
                
                # compare performance and save the best model
                for idx, metric in enumerate(self.args['eval_metric']): 
                    best_score[self.metric_dict[metric]] = self.compare_metric_save_model(metric, best_score[self.metric_dict[metric]], valid_score[self.metric_dict[metric]])
                                        
                start_step = step + 1
        self.writer.close()

        return best_score

    # Prediction
    def prediction(self, metric, test_loader):
        ''' Leave-one-subject-out, test the best accuracy model '''
        print("Testing...")
        
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.args['total_path'], 'models', metric ,"{}_bestmodel").format(
                self.flatten_subjectList[self.subject_id]), map_location=self.args['device']))
        except FileNotFoundError:
            print("No existing trained model")
            
        if self.args['cuda']: 
            self.model.cuda(device=self.args['device'])

        loss, acc, bacc, f1score, precision, roc_auc, recall, cost = self.eval("test", test_loader, self.args['steps']+1, 1)
        return loss, acc, bacc, f1score, precision, roc_auc, recall, cost
        
    '''
    ###########################################################################################
    #  Train
    ###########################################################################################
    '''
    def train(self, minibatches, step):
        self.model.train()
        
        data = torch.cat([x for x,_ in minibatches])
        target = torch.cat([y for _,y in minibatches])

        onehot_target = util.to_one_hot(target, self.args['n_classes']).to(self.args['device'])
        
        lossfn=torch.nn.CrossEntropyLoss()
        self.optimizer.zero_grad()
        
        
        if self.args["aug_type"]=="mixup":
            data, onehot_target=util.mixup_data(data, onehot_target)
        
        output, target=self.model(data, onehot_target) # forward
    
        loss = lossfn(output, target) 
        pred = output.argmax(dim=1, keepdim=True)            

        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss)

        if step % self.args['checkpoint_freq'] == 0: 
            l=sum(self.losses)/len(self.losses)
            self.losses=[]
            acc=accuracy_score(target.argmax(dim=1).cpu().numpy(), pred.cpu().numpy())
            print('Train Epoch: {}\t Loss: {:.4f}\t ACC: {:.4f}'.format(step+1, l, acc))
            self.write_tensorboard("train", step, acc=acc, loss=l)


    '''
    ###########################################################################################
    #  Evaluation
    ###########################################################################################
    '''    
    ## EVALUATE 
    def eval(self, phase, loader, step=0, t=0):
        self.model.eval()  
        test_loss = []
        lossfn = torch.nn.CrossEntropyLoss() 

        time_cost=[]
        with torch.no_grad(): 
            for idx, datas in enumerate(loader):
                s_time=time.time()
                data, target = datas[0].to(self.args['device']), datas[1].to(self.args['device'], dtype=torch.int64)

                outputs, _ = self.model(data)
                
                e_time=time.time()
                time_cost.append(e_time-s_time)
                test_loss.append(lossfn(outputs, target))

                if idx==0:
                    preds=outputs.argmax(dim=1,keepdim=False)
                    targets=target
                else:
                    preds=torch.cat([preds, outputs.argmax(dim=1, keepdim=False)]) 
                    targets=torch.cat([targets,target])
                    
        loss = sum(test_loss)/len(loader.dataset)
        
        targets=targets.cpu().numpy()
        preds=preds.cpu().numpy()

        acc=accuracy_score(targets, preds)
        bacc=balanced_accuracy_score(targets,preds)
        f1=f1_score(targets,preds)
        preci=precision_score(targets,preds)
        roc_auc=roc_auc_score(targets,preds)
        recall=recall_score(targets,preds)

        print(phase.capitalize(),'Loss: {:.4f}, Acc: {:.4f}%, Bal Acc: {:.4f}%, F1: {:.4f}, Preci: {:.4f}, Recall: {:.4f}'
                    .format(loss, acc, bacc, f1, preci, recall))

        self.write_tensorboard(phase, step, loss, acc, bacc, f1, preci, roc_auc, recall)

        if phase=="valid":
            return loss.item(), acc, bacc, f1
        elif phase=="test":
            return loss.item(), acc, bacc, f1, preci, roc_auc, recall, time_cost

    
    '''
    ###########################################################################################
    #  ETC.
    ###########################################################################################
    '''
    ############## compare valid_score and evaluation metric and save the best model ##################
    def compare_metric_save_model(self, eval_metric, best_score, valid_score):
        ## compare validation accuracy of this epoch with the best accuracy score
        if eval_metric=="loss":
            ## if validation loss <= best loss, then save model(.pt)
            if best_score >= valid_score:
                best_score = valid_score
                torch.save(self.model.state_dict(), os.path.join(self.args['total_path'], 'models',eval_metric,"{}_bestmodel".format(self.flatten_subjectList[self.subject_id])))
        else:
            ## if validation accuracy >= best accuracy, then save model(.pt)
            if valid_score >= best_score:
                best_score = valid_score
                torch.save(self.model.state_dict(), os.path.join(self.args['total_path'], 'models',eval_metric,"{}_bestmodel".format(self.flatten_subjectList[self.subject_id])))
        
        return best_score
    
    ########################### Tensorboard ###################
    def write_tensorboard(self, phase, step, loss=0, acc=0, bacc=0, f1=0, preci=0, roc_auc=0, sensitivity=0):
        if phase=='train':
            self.writer.add_scalar(f'{phase}/acc', acc, step)
            self.writer.add_scalar(f'{phase}/loss', loss, step)
        else:
            self.writer.add_scalar(f'{phase}/loss', loss, step)
            self.writer.add_scalar(f'{phase}/acc', acc, step)
            self.writer.add_scalar(f'{phase}/balanced_acc', bacc, step)  
            self.writer.add_scalar(f'{phase}/f1score', f1, step)
            self.writer.add_scalar(f'{phase}/precision', preci, step)
            self.writer.add_scalar(f'{phase}/roc_auc', roc_auc, step) 
            self.writer.add_scalar(f'{phase}/recall_sensitivity', sensitivity, step) 
    