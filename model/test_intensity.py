import os
import sys
import numpy as np
import cv2
import sklearn
from sklearn.metrics import accuracy_score, f1_score

#0 - debug, 1 - info (still a LOT of outputs), 2 - warnings, 3 - errors
os.environ['GLOG_minloglevel'] = '2'
sys.path.insert(0,'/code/caffe/python')
import caffe


def load_image(path, crop_offset, crop_size):
    img = cv2.imread(path,cv2.IMREAD_COLOR) # BGR
    img = img[crop_offset:crop_offset+crop_size, crop_offset:crop_offset+crop_size]
    im = (img-128.0)*0.0078125        
    return im

gpu = 0
caffe.set_mode_gpu()
caffe.set_device(gpu)

model_path = './'
data_name = 'BP4D'

img_path_prefix = '../imgs/'+data_name+'_aligned/'
au_net_model = model_path + 'deploy_intensity.prototxt'
start_iter = 1
n_iters = 12
batch_size = 8
crop_size = 176
crop_offset = 12

test_img_list = open('../'+data_name+'_test_path.txt').readlines()
img_num = len(test_img_list)

test_chunk_list = [test_img_list[i:i + batch_size] for i in xrange(0, len(test_img_list), batch_size)]
  
for _iter in range(start_iter, n_iters+1):
    au_net_weights = model_path + 'AUNet_iter_'+str(_iter)+'0000.caffemodel'
    au_net = caffe.Net(au_net_model, au_net_weights, caffe.TEST)
    transformer = caffe.io.Transformer({'data': au_net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    
    start = True
    for batch_ind, img_paths in enumerate(test_chunk_list):
        for ind, img_path in enumerate(img_paths):
            im = load_image(img_path_prefix+img_path.strip(), crop_offset, crop_size)       
            au_net.blobs['data'].data[ind] = transformer.preprocess('data', im)
        au_net.forward()    
        concat_intensity_final = au_net.blobs['concat_intensity_final'].data
        concat_intensity_final = np.array(concat_intensity_final)
        if start:
            AU_intensity_pred = concat_intensity_final
            start = False
        else:
            AU_intensity_pred = np.concatenate((AU_intensity_pred, concat_intensity_final))
        print _iter, batch_ind
            
    AU_intensity_pred = AU_intensity_pred[0:img_num,:]
    np.savetxt(model_path+data_name+'_test_AU_intensity_pred-'+str(_iter)+'_all_.txt', AU_intensity_pred, fmt='%f', delimiter='\t')