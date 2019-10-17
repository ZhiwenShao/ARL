#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/cosine_similarity_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void CosineSimilarityLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  Dtype loss = 0;
  const Dtype * pa = bottom[0]->gpu_data();
  const Dtype * pb = bottom[1]->gpu_data();
  for (int i=0; i<num; ++i) {
    Dtype dot;
    caffe_gpu_dot(dim, pa, pb, &dot);
    loss += dot;
    pa += dim; pb += dim;
  }
  loss = Dtype(1.)- loss/num;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CosineSimilarityLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;  
  for (int i=0; i<2; ++i) {
    if (propagate_down[i]) {
      const int index = (i == 0) ? 1 : 0;  
      for (int j=0; j<num; ++j) {
        caffe_gpu_scale(dim, -top[0]->cpu_diff()[0] / num,
          bottom[index]->gpu_data() + bottom[index]->offset(j),
          bottom[i]->mutable_gpu_diff() + bottom[i]->offset(j));   
      }  
    }  
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CosineSimilarityLossLayer);

}  // namespace caffe
