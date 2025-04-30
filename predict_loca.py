import os
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import utils  # your utils module with model_prediction
import splitting.dataset_loca as dataset  # updated dataset that returns both targets with proper mapping
from splitting.dataset_loca import loc_mapping


parser = argparse.ArgumentParser(description='PET lymphoma classification - Prediction with Localization')

# I/O PARAMS
parser.add_argument('--output', type=str, default='.', help='name of output folder (default: ".")')
parser.add_argument('--splits', type=int, default=None, nargs='+', help='data split range to predict for')

# MODEL PARAMS
parser.add_argument('--chptfolder', type=str, default='results', help='path to trained models (default: "results")')
parser.add_argument('--normalize', action='store_true', default=True, help='normalize images')
parser.add_argument('--nfeat', type=int, default=512, help='number of embedded features (default: 512)')

# NEW: Classifier architecture parameters (to match training)
parser.add_argument('--cls_arch', type=str, default='simple', choices=['simple', 'complex'],
                    help='Classifier architecture used during training')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='Hidden dimension for complex classifier (if used)')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate for complex classifier (if used)')

# TRAINING PARAMS (for prediction, mainly data loading)
parser.add_argument('--batch_size', type=int, default=100, help='batch size for prediction (default: 100)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')

def main():
    global args
    args = parser.parse_args()
    
    # Load best run information (from training)
    bestrun = pd.read_csv(os.path.join(args.chptfolder, 'best_run.csv'))
    
    if args.splits is not None:
        splits = range(args.splits[0], args.splits[1] + 1)
    else:
        splits = range(20)
    
    t0 = time.time()
    for split in splits:
        run = bestrun[bestrun.split == split].run.values[0]
        chpnt = os.path.join(args.chptfolder, 'checkpoint_split' + str(split) + '_run' + str(run) + '.pth')
        
        # Load the model from the checkpoint with the correct classifier parameters.
        model = utils.get_model_multitask()
        ch = torch.load(chpnt, weights_only=False)

        model_dict = model.state_dict()
        
        pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.cuda()
        cudnn.benchmark = True
        
        # Get the test dataset (using the localization dataset helper).
        _, _, _, dset, _ = dataset.get_datasets_singleview_withLoca(None, args.normalize, False, split)
        print('  dataset size: {}'.format(len(dset.df)))
        out = predict(dset, model)
        pred_name = 'pred_split' + str(split) + '_run' + str(run) + '.csv'
        out.to_csv(os.path.join(args.output, pred_name), index=False)
        print('  loop time: {:.0f} min'.format((time.time()-t0)/60))
        t0 = time.time()

def predict(dset, model):
    loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # Call the test routine which now returns three outputs:
    probs, feats, loca_preds = test(loader, model)
    out = dset.df.copy().reset_index(drop=True)
    out['probs'] = probs
    out['true_loc'] = out['target_loc'].map(loc_mapping)
    # Create a DataFrame for extracted features from the backbone.
    feats_df = pd.DataFrame(feats, columns=['F{}'.format(x) for x in range(1, args.nfeat+1)])
    # Create a DataFrame for the localization probabilities (14 classes)
    loca_cols = ['L{}'.format(x) for x in range(14)]
    loca_df = pd.DataFrame(loca_preds, columns=loca_cols)
    # Also add the predicted localization label (obtained via argmax).
    out['pred_loc'] = np.argmax(loca_preds, axis=1)
    out = pd.concat([out, feats_df, loca_df], axis=1)
    return out

def test(loader, model):
    model.eval()
    num_samples = len(loader.dataset)
    probs = np.empty(num_samples)
    feats = np.empty((num_samples, args.nfeat))
    loca_preds = np.empty((num_samples, 14))  # 14 localization classes
    with torch.no_grad():
        for i, (input, _) in enumerate(loader):
            input = input.cuda()
            # For the multitask model, forward_feat returns (features, output_class, output_loc)
            f, output_class, output_loc = model.forward_feat(input)
            output_class = F.softmax(output_class, dim=1)
            output_loc = F.softmax(output_loc, dim=1)
            batch_size = input.size(0)
            start = i * args.batch_size
            end = start + batch_size
            feats[start:end, :] = f.detach().cpu().numpy()
            probs[start:end] = output_class.detach().cpu().numpy()[:, 1]
            loca_preds[start:end, :] = output_loc.detach().cpu().numpy()
    return probs, feats, loca_preds

if __name__ == '__main__':
    main()
