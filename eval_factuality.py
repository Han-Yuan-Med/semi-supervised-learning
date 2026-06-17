# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset
import json
from utils_metrics import compute_metrics
from train import get_config
from semilearn.datasets import get_collactor
from transformers import AutoTokenizer, AutoModelForSequenceClassification

lb_to_id_bymodel = {
            "roberta": {"contradiction": 0, "neutral": 1, "entailment": 2},
            "bart": {"contradiction": 0, "neutral": 1, "entailment": 2},
            "deberta": {"contradiction": 0, "entailment": 1, "neutral": 2},
            "modernbart": {"entailment": 0, "neutral": 1, "contradiction": 2}
        }
id_to_lb_bymodel = {k:{v1:k1 for k1, v1 in v.items()} for k, v in lb_to_id_bymodel.items()} 
selected_classes = ["entailment", "contradiction"]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def create_experiment_dir(args) -> str:
    """Create a structured directory for the experiment results"""
    # Create timestamp-based directory name
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Create directory structure
    exp_dir = os.path.join(
        args.output_dir,
        f"{args.net}_{args.algorithm}",
        f"labeled_ratio_{args.labeled_ratio:.2f}_seed_{args.seed}",
        f"fold_{args.fold}",
        "20250603"
    )
    
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

if __name__ == "__main__":
    args = get_config()
#     import argparse
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--load_path', type=str, required=True)

#     '''
#     Backbone Net Configurations
#     '''
#     parser.add_argument('--net', type=str, default='bert_base_uncased')
#     parser.add_argument('--net_from_name', type=bool, default=False)
#     parser.add_argument('--algorithm', type=str, default='freematch')
#     '''
#     Data Configurations
#     '''
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--data_dir', type=str, default='./data')
#     parser.add_argument('--dataset', type=str, default='nli')
#     parser.add_argument('--num_classes', type=int, default=2)

#     parser.add_argument('--labeled_ratio', type=float, required=True,
#                       help='Ratio of labeled data used for training (e.g., 0.1 for 10%)')
#     parser.add_argument('--seed', type=int, default=42,
#                       help='Random seed used for training')
#     parser.add_argument('--fold', type=int, required=True,
#                       help='Current fold number (0-based)')
    
#     parser.add_argument('--output_dir', type=str, default='evaluation_results',
#                       help='Base directory to save evaluation results')
    
#     args = parser.parse_args()
    args.output_dir = 'evaluation_results'
    
    exp_dir = create_experiment_dir(args)
    
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    print(f"***see what in checkpoint: {checkpoint.keys()}")
    # load_model = checkpoint['ema_model']
    load_model = checkpoint["model"]
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''
    
    net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes) # directly output 2-cat
    # print(f"what is net: {type(net)}")
    # print(f"what is ema_model: {checkpoint['ema_model'].keys()}")
    keys = net.load_state_dict(load_state_dict)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    # specify these arguments manually 
    args.num_labels = 1
    args.ulb_num_labels = 49600
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1

    dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes, args.data_dir, False)
    eval_dset = dataset_dict['eval']
    collact_fn = get_collactor(args, args.net)
    print(f"is the dataset okay? {eval_dset}")
    # eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)
    eval_loader = DataLoader(
        eval_dset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collact_fn, 
        pin_memory=False, 
        generator=None, 
        drop_last=False
    )
    
    acc = 0.0
    test_feats = []
    test_preds = []
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for data in eval_loader:
            x = data['x_lb']
            target = data['y_lb']

            if isinstance(x, dict):
                x_to = {k: v.cuda(args.gpu) for k, v in x.items()}
            logits = net(x_to)["logits"]
            # logits = logits / 0.8
            
            # check
            probs = logits.softmax(dim=-1)
            test_probs.append(probs.float().cpu().numpy())
            test_labels.append(target.float().cpu().numpy())

    test_probs = np.concatenate(test_probs)
    test_labels = np.concatenate(test_labels)
    
    # Save results
    test_labels = np.argmax(test_labels, axis=1)
    test_probs = test_probs[:, 1]
    results = compute_metrics(test_labels, test_probs) 
    results["prob"] = test_probs
    results["label"] = test_labels
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4, cls=NpEncoder)