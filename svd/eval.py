import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as fu

from pathlib import Path
from argparse import ArgumentParser
from svd.utils.labels import create_aligned_targets
from torch.nn.modules.utils import _pair, _quadruple


def opts_parser():
    parser = ArgumentParser(description='Evaluates singing voice predictions against ground truth.')
    parser.add_argument('infile', metavar='INFILE', type=str,
                        help='File to load the prediction curves from (.npz/.pkl format). '
                             'If given multiple times, prediction curves will be averaged.')
    parser.add_argument('--dataset', type=str, default='jamendo',
                        help='Name of the dataset to use (default: %(default)s)')
    parser.add_argument('--dataset-path', metavar='DIR', type=str, default=None, required=True,
                        help='Path to data of dataset which will be used')
    parser.add_argument('--threshold', type=float, default=None,
                        help='If given, use this threshold instead of optimizing it on the validation set.')
    parser.add_argument('--auroc', action='store_true', default=False,
                        help='If given, compute AUROC on the test set.')
    parser.add_argument('--smooth-width', metavar='WIDTH', type=int, default=56,
                        help='Apply temporal smoothing over WIDTH frames (default: (default)s)')
    parser.add_argument('--print_markdown', action='store_true', default=False,
                        help='Print results as markdown table instead of nice formatting.')
    return parser


def load_labels(files, predictions, fps, label_path):
    """ Loads labels for given predictions. """
    labels = []
    for file in files:
        with open(label_path / (file.name.rsplit('.', 1)[0] + '.lab')) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing') for start, end, label in segments]
        timestamps = torch.arange(len(predictions[file])) / float(fps)
        labels.append(create_aligned_targets(segments, timestamps, torch.bool))
    return labels


class MedianPool(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size, padding=0, same=False):
        super(MedianPool, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(1)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pr = pw // 2
            pl = pw - pr
            pb = ph // 2
            pt = ph - pb
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = fu.pad(x, self._padding(x), mode='replicate')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,))
        x = x.kthvalue(k=x.size(-1) // 2 + 1, dim=-1)[0]
        return x


def evaluate(predictions, truth, thres=None, smoothen=56, collapse_files=True, comp_auroc=False):
    """ Compute evaluation metrics given predictions and ground-truth. """
    device = predictions[0].device

    # smooth predictions if necessary
    if smoothen:
        median_pool = MedianPool(kernel_size=(1, smoothen), same=True)
        predictions = [median_pool(pred.view(1, 1, 1, -1)).squeeze() if len(pred) > 1 else pred for pred in predictions]

    # evaluate
    if thres is None or comp_auroc:
        thresholds = torch.cat((torch.tensor([1e-5, 1e-4, 1e-3]), torch.arange(1, 100) / 100.,
                                torch.tensor([1 - 1e-3, 1 - 1e-4, 1 - 1e-5])))
        if (thres is not None) and (thres not in thresholds):
            idx = torch.searchsorted(thresholds, thres)
            thresholds = torch.cat((thresholds[:idx], torch.tensor([thres]), thresholds[idx:]), dim=0)
    else:
        thresholds = torch.tensor([thres])
    thresholds = thresholds.to(device)
    tp = torch.zeros((len(predictions), len(thresholds)), dtype=torch.int).to(device)
    fp = torch.zeros((len(predictions), len(thresholds)), dtype=torch.int).to(device)
    tn = torch.zeros((len(predictions), len(thresholds)), dtype=torch.int).to(device)
    fn = torch.zeros((len(predictions), len(thresholds)), dtype=torch.int).to(device)

    for idx in range(len(predictions)):
        preds = predictions[idx] > thresholds[:, None]
        target = truth[idx].bool().view(-1, 1)
        nopreds = ~preds
        correct = (preds.T == target).T
        incorrect = ~correct
        tp[idx] = (correct * preds).sum(dim=1)
        fp[idx] = (incorrect * preds).sum(dim=1)
        tn[idx] = (correct * nopreds).sum(dim=1)
        fn[idx] = (incorrect * nopreds).sum(dim=1)
    if collapse_files:
        # treat all files as a single long file, rather than
        # averaging over file-wise results afterwards
        tp = tp.sum(dim=0, keepdim=True)
        fp = fp.sum(dim=0, keepdim=True)
        tn = tn.sum(dim=0, keepdim=True)
        fn = fn.sum(dim=0, keepdim=True)

    def save_div(a, b):
        b[a == 0] = 1
        return a / b.float()

    accuracy = (tp + tn) / (tp + fp + tn + fn).float()
    precision = save_div(tp, tp + fp)
    recall = save_div(tp, tp + fn)
    specificity = save_div(tn, tn + fp)
    fscore = save_div(2 * precision * recall, precision + recall)
    if comp_auroc:
        if not collapse_files:
            raise NotImplementedError("Sorry, we didn't need this so far.")
        rec = torch.cat((torch.tensor([0]), recall.squeeze().flip(0), torch.tensor([1])))
        one_minus_spec = torch.cat((torch.tensor([0]), 1 - specificity.squeeze().flip(0), torch.tensor([1])))
        auroc = torch.trapz(rec, one_minus_spec)
    else:
        auroc = float('nan')
    if thres is None:
        best = torch.argmax(accuracy.mean(dim=0))
    else:
        best = torch.searchsorted(thresholds, thres)
    return thresholds[best], {
        'accuracy': accuracy[:, best],
        'precision': precision[:, best],
        'recall': recall[:, best],
        'specificity': specificity[:, best],
        'fscore': fscore[:, best],
        'auroc': auroc,
    }


def run_evaluation(preds, file_list_val, file_list_test, options, fps, label_path):
    """ Perform evaluation and output result. """
    # optimize threshold on validation set if needed
    if options.threshold is None:
        options.threshold, _ = evaluate([preds[fn].squeeze() for fn in file_list_val],
                                        load_labels(file_list_val, preds, fps, label_path),
                                        smoothen=options.smooth_width)

    # evaluate on test set
    threshold, results = evaluate([preds[fn].squeeze() for fn in file_list_test],
                                  load_labels(file_list_test, preds, fps, label_path),
                                  smoothen=options.smooth_width,  thres=options.threshold, comp_auroc=options.auroc)

    if options.print_markdown:
        print('| threshold | Precision | Recall | Specificity | F1 | error | AUROC |')
        auroc = np.nan
        if options.auroc:
            auroc = results['auroc'] * 100
        print(' %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | ' % (
            threshold.item(), results['precision'].mean().item() * 100, results['recall'].mean().item() * 100,
            results['specificity'].mean().item() * 100, results['fscore'].mean().item() * 100,
            (1 - results['accuracy'].mean().item()) * 100, auroc
        ))
    else:
        print('thr: %.2f, prec: %.3f, rec: %.3f, spec: %.3f, f1: %.3f, err: %.3f' % (
            threshold.item(), results['precision'].mean().item() * 100, results['recall'].mean().item() * 100,
            results['specificity'].mean().item() * 100, results['fscore'].mean().item() * 100,
            (1 - results['accuracy'].mean().item()) * 100))
        if options.auroc:
            print('auroc: %.3f' % (results['auroc'] * 100))


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    infile = options.infile
    fps = 70

    # prepare paths and files
    data_path = Path(options.dataset_path)
    info_path = Path(__file__).parent.parent / 'dataset' / options.dataset
    label_path = info_path / 'labels'
    with open(str(info_path / 'filelists' / 'valid')) as f:
        file_list_val = [data_path / l.rstrip() for l in f if l.rstrip()]
    with open(str(info_path / 'filelists' / 'test')) as f:
        file_list_test = [data_path / l.rstrip() for l in f if l.rstrip()]

    # load network predictions
    preds = np.load(infile)
    preds_ = {Path(f): torch.tensor(preds[f]) for f in preds.keys()}

    run_evaluation(preds_, file_list_val, file_list_test, options, fps, label_path)


if __name__ == '__main__':
    main()
