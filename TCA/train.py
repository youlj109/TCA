# coding=utf-8
import os
import sys
import time
import numpy as np
import argparse
import timm
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ
from datautil.getdataloader import get_img_dataloader

def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam hyper-param')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=3, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='PACS')
    parser.add_argument('--data_dir', type=str, default='../data/PACS', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='1', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lam', type=float,
                        default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=50, help="max iterations")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet50',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase,ViT-B16/32,ViT-L16/32,ViT-H14")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--opt_type',type=str,default='Adam')  #if want to use SGD, please set SGD
    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)
    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_img_dataloader(args)
    
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()

    algorithm.train()
    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)
    
    if args.dataset in ['CIFAR-10', 'CIFAR-100']:
        class Divided_module(nn.Module):
            def __init__(self, args):
                super(Divided_module, self).__init__()
                self.args = args
                self.algorithm_res = self._load_algorithm()
                if 'ViT' in self.args.net:
                    self.algorithm_res.head = nn.Identity()    
                    dummy_input = torch.zeros(1, 3, 224, 224)  
                    output = self.algorithm_res(dummy_input)   
                    self.num_ftrs = output.shape[1]    
                    self.featurizer = nn.Sequential(self.algorithm_res, nn.Flatten()) 
                else:
                    self.num_ftrs = self.algorithm_res.fc.in_features  
                    self.algorithm_res.fc = nn.Linear(self.num_ftrs, args.num_classes) 
                    self.featurizer = nn.Sequential(*list(self.algorithm_res.children())[:-1],nn.Flatten())

                self.classifier = nn.Linear(self.num_ftrs, self.args.num_classes)
                self.network = nn.Sequential(self.featurizer, self.classifier)
                
            def _load_algorithm(self):
                if self.args.net == 'resnet50':
                    return models.resnet50()
                elif self.args.net == 'resnet18':
                    return models.resnet18()
                elif self.args.net == 'ViT-B16':
                    return timm.create_model('vit_base_patch16_224_in21k',pretrained=True,num_classes=0)
                else:
                    print('Net selected wrong!')
                    return None

            def forward(self, x):
                return self.network(x)
            
            def predict(self, x):
                return self.network(x)
        
        algorithm = Divided_module(args)
        criterion_C = nn.CrossEntropyLoss()
        if args.opt_type=='SGD':
            optimizer_C = optim.SGD(algorithm.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        elif args.opt_type=='Adam':
            optimizer_C = optim.Adam(algorithm.parameters(), lr=args.lr)
        device_C = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        algorithm.train()
       
    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)
    acc_record = {}
    acc_type_list = ['valid','target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0
    print('===========start training===========')
    sss = time.time()
    for epoch in range(args.max_epoch):
        if args.dataset in ['CIFAR-10', 'CIFAR-100']:
            running_loss = 0.0
            for i, data in enumerate(train_loaders, 0):
                inputs, labels = data[0].to(device_C), data[1].to(device_C)
                algorithm = algorithm.to(device_C)
                optimizer_C.zero_grad()
                outputs = algorithm.predict(inputs)
                loss = criterion_C(outputs, labels)
                loss.backward()
                optimizer_C.step()
                running_loss += loss.item()
            step_vals = running_loss/len(train_loaders)
        else:
            for iter_num in range(args.steps_per_epoch):
                minibatches_device = [(data)
                                    for data in next(train_minibatches_iterator)]
                if args.algorithm == 'VREx' and algorithm.update_count == args.anneal_iters:
                    opt = get_optimizer(algorithm, args)
                    sch = get_scheduler(opt, args)
                step_vals = algorithm.update(minibatches_device, opt, sch)


        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):
            print('===========epoch %d===========' % (epoch))
            s = ''
            for item in loss_list:
                if args.dataset in ['CIFAR-10', 'CIFAR-100']:
                    s += (item+'_loss:%.4f,' % step_vals)
                else:
                    s += (item+'_loss:%.4f,' % step_vals[item])
            print(s[:-1])
            s = ''
            if args.dataset in ['CIFAR-10', 'CIFAR-100']:
                acc_record['valid'] = np.mean(np.array(modelopera.accuracy_C(algorithm, eval_loaders)))
                s += ('valid'+'_acc:%.4f,' % acc_record['valid'])
                if acc_record['valid'] > best_valid_acc:
                    best_valid_acc = acc_record['valid']
                    save_checkpoint('model.pkl', algorithm, args)
            else:
                for item in acc_type_list:
                    acc_record[item] = np.mean(np.array([modelopera.accuracy(
                        algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
                    s += (item+'_acc:%.4f,' % acc_record[item])
                if acc_record['valid'] > best_valid_acc:
                    best_valid_acc = acc_record['valid']
                    target_acc = acc_record['target']
                    save_checkpoint('model.pkl', algorithm, args)
                print(s[:-1])
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args)
            print('total cost time: %.4f' % (time.time()-sss))
            algorithm_dict = algorithm.state_dict()

    save_checkpoint('model_last.pkl', algorithm, args)

    print('valid acc: %.4f' % best_valid_acc)
    print('DG result: %.4f' % target_acc)

    with open(os.path.join(args.output, 'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time()-sss)))
        f.write('valid acc:%.4f\n' % (best_valid_acc))
        f.write('target acc:%.4f' % (target_acc))

   