import os
import json
import sys
import math
import torch
import _init_paths
from parse_args import parse_args
from AlexNet import *
from Baseline import *
from shared.data_process import kill_data_processes
from train_utils import model_at, parse_experiment, metrics, train, test, \
   data_setup, set_seed, create_optimizer, check_overwrite, resume

def save_model(args, epoch):
    model_bk = args.model
    args.model = ''
    torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model_bk.state_dict(),
                'optimizer': args.optimizer.state_dict()},
            os.path.join(args.odir, 'models/model_%d.pth.tar' % (epoch + 1)))
    args.model = model_bk


def main():
    args = parse_args()
    set_seed(args.seed)
    
    #set up train dataloader
    train_data_queue, train_data_processes = data_setup(args, 'train', args.nworkers,
                                                        repeat=True)
    eval(args.net + '_setup')(args)

    
    # Create model and optimizer
    if args.resume or args.eval:
        #reload model and resume training if model exists
        last_epoch, best_epoch, best_val_loss, num_params = parse_experiment(args.odir)
        i = last_epoch
        if args.eval:
            i = best_epoch
        args.resume = model_at(args, i)
        model, args.optimizer, stats, r_args = resume(args, i)
    else:
        #create new model if model does not exists
        check_overwrite(os.path.join(args.odir, 'trainlog.txt'))
        model = eval(args.net + '_create_model')(args)
        args.optimizer = create_optimizer(args, model)
        args.start_epoch = 0
        stats = []
    
    print('Will save to ' + args.odir)
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    if not os.path.exists(args.odir + '/models'):
        os.makedirs(args.odir + '/models')
    with open(os.path.join(args.odir, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(["'"+a+"'" if (len(a)==0 or a[0]!='-') else a for a in sys.argv]))

    print(model)
    args.model = model
    args.step = eval(args.net + '_step') #step function for given model

    # Training loop
    epoch = args.start_epoch
    if args.eval == 0:
        for epoch in range(args.start_epoch, args.epochs):
            print('Epoch {}/{} ({}):'.format(epoch + 1, args.epochs, args.odir))
            loss_names, losses = train(args, epoch, train_data_queue, train_data_processes)
            ldict = {'epoch': epoch + 1}
            if (epoch+1) % args.test_nth_epoch == 0 or epoch+1==args.epochs:
                #test model on validation set (or training set if overfitting) and save losses
                loss_names, losses_val = test(args.test_split, args)
                for ix, loss in enumerate(loss_names):
                    print('-> Train {}: {}, \t Val {}: {}'.format(loss_names[ix], losses[ix], loss_names[ix], losses_val[ix]))
                    ldict[loss_names[ix] + '_train'] = losses[ix]
                    ldict[loss_names[ix] + '_val'] = losses_val[ix]
            else:
                # just train, no testing
                for ix, loss in enumerate(loss_names):
                    print('-> Train {}: {}'.format(loss_names[ix], losses[ix]))
                    ldict[loss_names[ix] + '_train'] = losses[ix]
            stats.append(ldict)

            # save model
            if (epoch+1) % args.save_nth_epoch == 0 or epoch+1==args.epochs:
                save_model(args, epoch)
            
            # evaluate 
            if (epoch+1) % args.test_nth_epoch == 0 and epoch+1 < args.epochs:
                #test model on validation set (or training set if overfitting) and save metrics
                split = args.test_split
                metrics(split, args, epoch)
                with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
                    json.dump(stats, outfile)

            if math.isnan(losses[0]): break

        if len(stats)>0:
            with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
                json.dump(stats, outfile)

    kill_data_processes(train_data_queue, train_data_processes)

    """
    #evaluate on test set
    split = 'test'
    if args.eval:
        split = 'test'
    metrics(split, args, epoch)
    """
    

if __name__ == '__main__':
    main()