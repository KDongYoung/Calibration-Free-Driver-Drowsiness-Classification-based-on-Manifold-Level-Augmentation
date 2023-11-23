import argparse
import os

""" Experiment Setting """ 
# ARGUMENT
parser = argparse.ArgumentParser(description='Fatigue') 
parser.add_argument('--data_root', default='DATASET_DIR/', help="name of the data folder")
parser.add_argument('--run_code_folder', default='')
parser.add_argument('--save_root', default='./MODEL_SAVE_DIR/', help="where to save the models and tensorboard records")
parser.add_argument('--result_dir', default="", help="save folder name")
parser.add_argument('--total_path', default="", help='total result path')
parser.add_argument('--cuda', type=bool, default=True, help='cuda')
parser.add_argument('--cuda_num', type=int, default=0, help='cuda number')
parser.add_argument('--device', default="", help='device')

parser.add_argument('--n_classes', type=int, default=2, help='num classes')
parser.add_argument('--n_channels', type=int, default=30, help='num channels')
parser.add_argument('--n_timewindow', type=int, default=384, help='timewindow')

parser.add_argument('--res_layer', type=int, default=[], help='after which residual layer is MixStyle used')
parser.add_argument('--track_running', type=bool, default=True, help='track mean, var of batchnorm') 
parser.add_argument('--mixup_alpha', type=int, default=0, help='alpha parameter for mixup')
parser.add_argument('--mixstyle_prob', default=0.5, help='probability of apply mixstyle 50% apply -> 0.5') 

parser.add_argument('--optimizer', default="Adam", help='optimizer')
parser.add_argument('--lr', type=float, default=0, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0, help='the amount of weight decay in optimizer') 
parser.add_argument('--scheduler', default="CosineAnnealingLR", help='scheduler')
parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size of each subject for training (default: 16)')
parser.add_argument('--valid-batch-size', type=int, default=1, metavar='N', help='valid batch size for training (default: 1)') 
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for ONLINE testing (default: 1)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number worker')

parser.add_argument('--steps', type=int, default=0, help='Number of steps')
parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint every N steps')
parser.add_argument('--seed', type=int, default=2023, help='seed')

parser.add_argument('--model_name', default='resnet8', help='trained model name')
parser.add_argument('--mode', default='train', help='train, infer')
parser.add_argument('--aug_type', default='', help='mixstyle, mixup, manifoldmixup')

parser.add_argument('--eval_metric', default=['loss', 'acc', 'bacc', 'f1'], help='evaluation metric for model selection')
parser.add_argument('--metric_dict', default={"loss":0,"acc":1, "bacc":2, "f1":3}, help='total evaluation metric')

args = parser.parse_args()
args=vars(args)

subjectList=['S01', 'S02', 'S03', 'S04', 
             'S05', 'S06', 'S07', 'S08', 
             'S09', 'S10', 'S11'] 

model_name='Resnet18'

args["run_code_folder"]=os.path.realpath(__file__) # current folder name of running code
args["n_classes"]=2
args["n_channels"]=30
args["n_timewindow"]=384
args["lr"]=0.002 
args["weight_decay"]=0 
args["steps"]=3000
args['optimizer']="Adam" 
args["data_root"]=args["data_root"]+"/"


import Main

# mixup augmentaion
args["aug_type"]="mixup"
args['mixup_alpha']=0.5
Main.main(subjectList, args, model_name) 

# manifold mixup augmentaion
args["aug_type"]="manifoldmixup"
args['mixup_alpha']=0.5
Main.main(subjectList, args, model_name) 

# mixstyle augmentation
args["aug_type"]="mixstyle"
args["res_layer"]=[1,2,3,4]
Main.main(subjectList, args, model_name) 
