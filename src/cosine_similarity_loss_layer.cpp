#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/cosine_similarity_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void CosineSimilarityLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
}
  
template <typename Dtype>
void CosineSimilarityLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  Dtype loss = 0;
  const Dtype * pa = bottom[0]->cpu_data();
  const Dtype * pb = bottom[1]->cpu_data();
  for (int i=0; i<num; ++i) {
    loss += caffe_cpu_dot(dim, pa, pb);
    pa += dim; pb += dim;
  }
  loss = Dtype(1.)- loss/num;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CosineSimilarityLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int index = (i == 0) ? 1 : 0;  
      for (int j=0; j<num; ++j) {
        caffe_cpu_scale(dim, -top[0]->cpu_diff()[0] / num,
          bottom[index]->cpu_data() + bottom[index]->offset(j),
          bottom[i]->mutable_cpu_diff() + bottom[i]->offset(j));
      }
    }
  }   
}

#ifdef CPU_ONLY
STUB_GPU(CosineSimilarityLossLayer);
#endif

INSTANTIATE_CLASS(CosineSimilarityLossLayer);
REGISTER_LAYER_CLASS(CosineSimilarityLoss);

}  // namespace caffe
