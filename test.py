from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Union, Literal, List
from functools import partial

import numpy as np
import cv2
from adeval import EvalAccumulatorCuda
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch


@dataclass
class Item:
    anomap: Tensor
    score: float
    mask: Tensor
    label: int
    path: str


class DemoDataset(Dataset[Dict[
    Literal['anomap', 'score', 'mask', 'label', 'path'], Union[Tensor, float, int]
]]):

    def __init__(self, result: Dict[str, Tuple[np.ndarray, float, Optional[str]]],
                 eval_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.result = result
        self.keys = sorted(result.keys())
        self.eval_size = eval_size

    def __getitem__(self, index: int) -> Item:
        assert isinstance(index, int) and -len(self) <= index < len(self)
        path = self.keys[index]
        anomap, score, mask_path = self.result[path]
        mask, label = get_gt(path if self.eval_size is None else self.eval_size, mask_path)

        anomap = cv2.resize(anomap, mask.shape[0:2][::-1], interpolation=cv2.INTER_LINEAR)

        return asdict(Item(anomap=torch.from_numpy(anomap),
                           score=score,
                           mask=torch.from_numpy(mask),
                           label=label,
                           path=path))

    def __len__(self) -> int:
        return len(self.keys)


@dataclass
class _Args:
    anomap: str
    remove_prefix: str
    add_prefix: str
    evaluate_size: Optional[Tuple[int, int]] = None
    sample_key_pat: Optional[str] = None
    expected_views: Optional[int] = None
    more_anomap: Tuple[str, ...] = tuple()

    def __reduce__(self):
        return partial(type, '_Args', (object,)), (asdict(self),)


def parse_args() -> _Args:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('anomap', type=str, help='''
path to a pickle of a dict[str, tuple[ndarray, float, str | None]], whose key is path to image
of dataset, and values are predicted anomap, predicted image-level score and path to the ground 
truth anomaly mask respectively
                        ''')
    parser.add_argument('--remove_prefix', type=str, default=None,
                        help='remove prefix from path to image & mask first')
    parser.add_argument('--add_prefix', type=str, default=None,
                        help='add another prefix to path to image & mask')
    parser.add_argument('--evaluate_size', type=str, default=None,
                        help='''calculate metrics at specific size rather than original size of
                        images, can be [size] (a single integer) or a [w,h] (a tuple of
                        integers)''')
    parser.add_argument('--sample_key_pat', type=str, default=None,
                        help='regex to capture the sample id from image path, '
                        'for evaluating sample-level metrics')
    parser.add_argument('--expected_views_per_sample', type=int, default=None,
                        help='if given, script will check if each sample '
                        'has the expected views, and raise error if not')
    parser.add_argument('-a', '--more_anomap', type=str, action='append', default=[])
    
    args = parser.parse_args()

    evaluate_size = None
    if args.evaluate_size is not None:
        try:
            size = int(args.evaluate_size)
            evaluate_size = (size, size)
        except ValueError:
            evaluate_size = tuple(int(val) for val in args.evaluate_size.split(','))
            if len(evaluate_size) != 2:
                raise ValueError('evaluate_size must be integer or 2-d tuple of integers')

    return _Args(anomap=args.anomap,
                 remove_prefix=args.remove_prefix,
                 add_prefix=args.add_prefix,
                 evaluate_size=evaluate_size,
                 sample_key_pat=args.sample_key_pat,
                 expected_views=args.expected_views_per_sample,
                 more_anomap=tuple(args.more_anomap))


def load_anomap(path: str) -> Dict[str, Tuple[np.ndarray, float, Optional[str]]]:
    import pickle
    from numbers import Real

    with open(path, 'rb') as fp:
        anomap_set = pickle.load(fp)

    assert isinstance(anomap_set, dict)
    assert all(isinstance(key, str) for key in anomap_set.keys())
    assert all(isinstance(anomap, np.ndarray) and anomap.ndim == 2 and
               isinstance(score, Real) and not np.isnan(score) and
               isinstance(mask, (str, None.__class__))
               for anomap, score, mask in anomap_set.values())

    return anomap_set


def get_gt(path: Union[str, Tuple[int, int]], mask: Optional[str]) -> Tuple[np.ndarray, int]:
    import cv2
    from PIL import Image

    w, h = Image.open(path).size if isinstance(path, str) else path
    if mask is None:
        # succ, ret = cv2.imencode('.png', np.zeros((h, w), dtype=np.uint8))
        # assert succ
        ret = np.zeros((h, w), dtype=np.uint8)
        label = 0
    else:
        data = np.fromfile(mask, dtype=np.uint8)
        maskimg = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        assert maskimg is not None
        if maskimg.shape != (h, w):
            maskimg = cv2.resize(maskimg, (w, h), interpolation=cv2.INTER_NEAREST)
        maskimg = np.where(
            maskimg > 127, np.full((), 255, dtype=np.uint8), np.zeros((), dtype=np.uint8)
        )
        label = 1 if np.any(maskimg) else 0
        # succ, ret = cv2.imencode('.png', maskimg)
        # assert succ
        ret = maskimg
    return ret, label


def main():
    import re
    from itertools import groupby
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing.context import SpawnContext
    import os
    try:
        import torch
    except ImportError:
        torch = None


    args = parse_args()
    sample_pat = re.compile(args.sample_key_pat) \
        if args.sample_key_pat is not None else None
 
    def replace(path: Optional[str]) -> Optional[str]:
        if path is None:
            return path
        if args.remove_prefix is not None:
            assert path.startswith(args.remove_prefix)
            path = path[len(args.remove_prefix):]
        if args.add_prefix is not None:
            path = args.add_prefix + path
        return path

    if sample_pat is not None:
        if args.expected_views is not None:
            assert args.expected_views > 0

        def get_sample_id(path: str) -> str:
            mat = sample_pat.search(path)
            assert mat is not None, (path, sample_pat)
            assert len(mat.groups()) == 1
            return mat.groups()[0]
    else:
        def get_sample_id(path: str) -> str:
            return ''

    result = load_anomap(args.anomap)
    _ = get_sample_id(next(iter(result.keys())))

    result = {replace(path): (anomap, score, replace(mask_path))
              for path, (anomap, score, mask_path) in result.items()}

    score_min = min(score for _, (_, score, _) in result.items())
    score_max = max(score for _, (_, score, _) in result.items())
    anomap_min = min(anomap.min() for _, (anomap, _, _) in result.items())
    anomap_max = max(anomap.max() for _, (anomap, _, _) in result.items())

    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max)
    print(score_min, score_max)
    print(anomap_min, anomap_max)
    image_scores: List[Tuple[str, float, int]] = []
    for items in DataLoader(DemoDataset(result, args.evaluate_size),
                            batch_size=16,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=8,
                            multiprocessing_context=SpawnContext()):
        anomap: Tensor = items['anomap']
        score: Tensor = items['score']
        mask: Tensor = items['mask']
        label: Tensor = items['label']
        path: List[str] = items['path']

        anomap = anomap.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        accum.add_anomap_batch(anomap, mask)
        accum.add_image(score, label)
        image_scores += [(key, img_score.item(), img_label.item())
                         for key, img_score, img_label in zip(path, score.reshape(-1), label.reshape(-1))]

    if sample_pat is not None:
        sample_pairs = [([score for _, score, _ in views],
                         [label for _, _, label in views],
                         [path for path, _, _ in views], key)
                        for key, views in [(key, list(views)) for key, views in groupby(
                             sorted(image_scores, key=lambda tup: tup[0]), key=lambda tup: get_sample_id(tup[0])
                        )]]
        if args.expected_views is not None:
            assert all(len(score) == args.expected_views for score, _, _, _ in sample_pairs), \
                [(score, label, path, key) for score, label, path, key in sample_pairs
                 if len(score) != args.expected_views]
        accum.add_sample([max(score) for score, _, _, _ in sample_pairs],
                         [max(label) for _, label, _, _ in sample_pairs])

    metrics = accum.summary()

    if sample_pat is not None:
        s_auroc, s_aupr = metrics['s_auroc'], metrics['s_aupr']
    else:
        s_auroc, s_aupr = float('nan'), float('nan')

    i_auroc, i_aupr = metrics['i_auroc'], metrics['i_aupr']
    p_auroc, p_ap, p_aupro = metrics['p_auroc'], metrics['p_aupr'], metrics['p_aupro']

    print(f'{args.anomap},'
          f's_auroc,{s_auroc},s_aupr,{s_aupr},'
          f'i_auroc,{i_auroc},i_aupr,{i_aupr},'
          f'p_auroc,{p_auroc},p_aupr,{p_ap},p_aupro,{p_aupro}', flush=True)

    if len(args.more_anomap) > 0:
        print('skip in testing')

if __name__ == '__main__':
    main()
