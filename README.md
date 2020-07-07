# Faster-RCNN
This repo follow by https://github.com/bubbliiiing/faster-rcnn-pytorch, I make it easy to study.
## File description
frcnn_all.py = stage1(rpn) +stage2(classifier)
You should deal with the coordinates of the propsals box(rois) and its corrections from stage2.
There's a problem here that I need to deal with is that how to understand DecodeBox(). But it didn't influence you predict the model.
## Model path
`https://drive.google.com/file/d/1bXLxg0jkZDZeYoOQARE4C1M4nnQGXtU3/view?usp=sharing`
## The next I will to do
1. Test the model on VOC datasets
2. Add the train code.


