# Instructions to use Mask-RCNN

## Input

Datasets of images you want to create masks.

## Output

Corresponding masks of images.

## Steps

1. Clone [Mask-RCNN](https://github.com/matterport/Mask_RCNN) in your own folder.
2. Copy `trans2mask.py` to `Mask_RCNN/samples`.
3. Modify the object labels and input directory in the code.
4. `python trans2mask.py` to run.