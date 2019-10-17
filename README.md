# ARL
This repository is the implementation of our paper "[Facial Action Unit Detection Using Attention and Relation Learning](https://arxiv.org/pdf/1808.03457.pdf)". The code is mainly borrowed from [JAA-Net](https://github.com/ZhiwenShao/JAANet).

# Getting Started
## Dependencies
- Dependencies for [Caffe](http://caffe.berkeleyvision.org/install_apt.html) are required

- The new implementations in the folders "src" and "include" should be merged into the official [Caffe](https://github.com/BVLC/caffe):
  - Add the .cpp, .cu files into "src/caffe/layers"
  - Add the .hpp files into "include/caffe/layers"
  - Add the content of "caffe.proto" into "src/caffe/proto"
  - Add "tools/convert_data.cpp" into "tools"
- New implementations used in our paper:
  - division_layer: divide a feature map into multiple identical subparts
  - combination_layer: combine mutiple sub feature maps
  - multi_stage_meanfield_au3 and meanfield_iteration: fully-connected conditional random field
  - sigmoid_cross_entropy_loss_layer: the weighting for the loss of each element is added
  - euclidean_loss_layer: used for AU intensity estimation: weighting the loss of each element, and setting the gradient as zero when the corresponding AU label is missing.
  - convert_data: convert the AU labels and weights to leveldb or lmdb
- Build Caffe

## Datasets
[BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and [DISFA](http://www.engr.du.edu/mmahoor/DISFA.htm)

The 3-fold partitions of both BP4D and DISFA can be found [here](https://github.com/ZhiwenShao/JAANet/tree/master/data)

## Preprocessing
- Prepare the training data
  - Run "prep/face_transform.cpp" to conduct similarity transformation for face images
  - Run "prep/combine2parts.m" to combine two partitions as a training set, respectively
  - Run "prep/write_AU_weight.m" to compute the weight of each AU for the training set
  - Run "tools/convert_imageset" of Caffe to convert the images to leveldb or lmdb
  - Run "tools/convert_data" to convert the AU labels and weights to leveldb or lmdb: the weights are shared by all the training samples (only one line needed)
  - Our method is evaluated by 3-fold cross validation. For example, “BP4D_combine_1_2” denotes the combination of partition 1 and partition 2
- Modify the train_val prototxt files:
  - Modify the paths of data
  - A recommended training strategy is that selecting a small set of training data for validation to choose a proper maximum iterations and then using all the training data to retrain the model

## Training on BP4D
```
cd model
sh train_net_bp4d.sh
```

## Training on DISFA
```
cd model
sh train_net_disfa.sh
```

## Testing
```
cd model
python test.py
```

## Citation
If you use this code for your research, please cite our paper
```
@article{shao2019facial,
  title={Facial action unit detection using attention and relation learning},
  author={Shao, Zhiwen and Liu, Zhilei and Cai, Jianfei and Wu, Yunsheng and Ma, Lizhuang},
  journal={IEEE Transactions on Affective Computing},
  year={2019},
  publisher={IEEE}
}
```
