# ADEval: Fast and Memory-efficient Routines for AUROC/AUPR/Pixel-AUPRO
via Iterative-Accumulating Algorithm with Optional CUDA Acceleration

## Requirement

- CPython 3.8 or above is recommanded, CPython 3.6 should be compatible but not tested

- Dependent packages
```shell
Pillow
opencv-python  # for ground-truth mask processing and PRO's CCA
numpy
torch  # Optional, for CUDA acceleration
```

- Optional dependent packages for comparing with reference implementation
```shell
scikit-learn
scipy
bisect
```

## Commandline Usage

- Quick Start
```shell
# example 1: evaluate result of object 'audiojack' from Real-IAD under Multi-view UIAD setting
python3 -m adeval --sample_key_pat "([a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_C[0-9]_" real_iad-audiojack-Mv_UIAD.pkl

# example 2: evaluate 4 results of object 'audiojack' from Real-IAD under FUAD settings (i.e. NR = 0.0, 0.1, 0.2, 0.4)
python3 -m adeval --sample_key_pat "([a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_C[0-9]_" \
  real_iad-audiojack-FUIAD0.pkl \
  -a real_iad-audiojack-FUIAD1.pkl \
  -a real_iad-audiojack-FUIAD2.pkl \
  -a real_iad-audiojack-FUIAD4.pkl
```

- Detailed Description of Arguments
```
usage: python3 -m adeval [-h] [--remove_prefix REMOVE_PREFIX] [--add_prefix ADD_PREFIX] [--sample_key_pat SAMPLE_KEY_PAT] [--expected_views_per_sample EXPECTED_VIEWS_PER_SAMPLE] [-a MORE_ANOMAP] anomap

positional arguments:
  anomap                path to a pickle of a dict[str, tuple[ndarray, float, str | None]], whose key is path to image
                        of dataset, and values are predicted anomap, predicted image-level score and path to the ground 
                        truth anomaly mask respectively

optional arguments:
  -h, --help            show this help message and exit
  --remove_prefix REMOVE_PREFIX
                        remove prefix from path to image & mask first
  --add_prefix ADD_PREFIX
                        add another prefix to path to image & mask
  --evaluate_size EVALUATE_SIZE
                        calculate metrics at specific size rather than original size of images, can be [size] (a single integer) or a [w,h] (a tuple of integers)
  --sample_key_pat SAMPLE_KEY_PAT
                        regex to capture the sample id from image path, for evaluating sample-level metrics
  --expected_views_per_sample EXPECTED_VIEWS_PER_SAMPLE
                        if given, script will check if each sample has the expected views, and raise error if not
  -a MORE_ANOMAP, --more_anomap MORE_ANOMAP
```

Here is a code snippet to construct anomap `.pkl` file that can be passed to ADEval commandline tool
```python
import numpy as np
import pickle

anomaps = {}
for image_path, mask_path in test_dataset:
    assert isinstance(image_path, str)
    assert mask_path is None or isinstance(mask_path, str)
    img = load_image(image_path)
    anomap, image_score = inference(img)
    assert isinstance(image_score, float)
    assert (isinstance(anomap, np.ndarray) and
            anomap.ndim == 2 and
            'float' in anomap.dtype.name)
    anomaps[image_path] = (anomap, image_score, mask_path)

with open('/path/to/save/anomap.pkl', 'wb') as fp:
    pickle.dump(anomaps, fp)
```


## API Usage

Following the code snippet below to utilize the iterative style API, so we do not have to
save ano-maps of all images in the test set.

```python
from adeval import EvalAccumulatorCuda
# for cpu-only environment, use the class below and remove all cuda-related
#    operation in the codes
from adeval import EvalAccumulator

def get_sample_id(path: str) -> str:
    ...
    # get id string of sample from path of image

def validation(...):

    # determin the lower & upper bound of image-level score according to your
    #   algorithm design, ensure that the scores of 99% of the images fall within 
    #   the given lower & upper bound
    score_min, score_max = ...
    # also determine the lower & upper bound of values in anomap, or set to 
    #   None to reuse bounds of image-level score
    anomap_min, anomap_max = ...
    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max)

    # the DataLoader of test set
    test_loader = ...
    image_scores: List[Tuple[str, float, int]] = []

    for items in test_loader:
        anomap, score = model(items['image'].cuda())
        mask: Tensor = items['mask']
        label: Tensor = items['label']
        path: List[str] = items['path']

        mask = mask.cuda(non_blocking=True)

        accum.add_anomap_batch(anomap, mask)
        accum.add_image(score, label)
        image_scores += [(key, img_score.item(), img_label.item())
                         for key, img_score, img_label in
                         zip(path, score.reshape(-1), label.reshape(-1))]

    sample_pairs = [([score for _, score, _ in views],
                     [label for _, _, label in views],
                     [path for path, _, _ in views], key)
                    for key, views in [(key, list(views)) for key, views in groupby(
                         sorted(image_scores, key=lambda tup: tup[0]), key=lambda tup: get_sample_id(tup[0])
                    )]]
    accum.add_sample([max(score) for score, _, _, _ in sample_pairs],
                     [max(label) for _, label, _, _ in sample_pairs])

    metrics = accum.summary()
    print(metrics)
    # contains:
    #   - sample-level auroc, aupr
    #   - image-level auroc, aupr
    #   - pixel-level auroc, aupr, aupro
```
And please refer to the functional example [test.py](test.py) for more details

For usage of direct APIs (i.e. `auroc_and_aupr()` etc.), please following implementation of [CLI](adeval/__main__.py)

## Acknowledgement
The team of [Real-IAD](https://realiad4ad.github.io/Real-IAD/) dataset has contributed numerous invaluable insights to this project.

The reference implementations are from repos [AnomalyDetection-SoftPatch](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch) and [open-iad](https://github.com/M-3LAB/open-iad):
- `metrics.py`: reference of AUROC & AUPR based on `scikit-learn` package from AnomalyDetection-SoftPatch
- `au_pro`: reference of AUPRO from open-iad
