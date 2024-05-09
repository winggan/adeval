## v1.1.0

### CLI Change
- Use `maskimg > 127` instead of `maskimg > 0` to binarize mask image, so we can correctly deal 
  with downsampled version of Real-IAD dataset (i.e. 1024, 512 or 256 version)
- Add `--nstrips` argument to adjust datasets with less than 1000 samples when calculating
  image/sample-level AUROC or AUPR

### API Change
- Expose `nstrips` parameter to adjust datasets with less than 1000 samples

## v1.0.0

First release