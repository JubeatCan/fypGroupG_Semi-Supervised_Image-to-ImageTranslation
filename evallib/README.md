# Instructions to evaluate

## Input

Two sets of images to evaluate
- With same dimensions
- And same number of images

## Output

KID and FID scores of the input images.

## Steps

1. Clone [KID and FID Evaluation](https://github.com/telecombcn-dl/2018-dlai-team1) in your own folder.
2. Copy `cal_inception.py` to root of that folder.
3. `python cal_inception.py --real_dataroot=<FirstSetImages> --fake_dataroot=<SecondSetImages> --kid_batch=<NumberOfImagesInKID> --fid_batch=<NumberOfImagesInFID>` to run.