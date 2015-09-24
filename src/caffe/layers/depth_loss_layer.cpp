#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void DepthLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
			<< "Inputs must have the same dimension.";
		diff_.ReshapeLike(*bottom[0]);
		d_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void DepthLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			int count = bottom[0]->count();
			Dtype* bottom1_log = new Dtype[count];

			//compute log for Y*
			for (int i = 0; i < count; i++)
			{
				bottom1_log[i] = log(bottom[1]->cpu_data()[i]);
				
			}

			//compute di
			caffe_sub(
				count,
				bottom[0]->cpu_data(),
				bottom1_log,
				d_.mutable_cpu_data());

			//part1 of Loss
			Dtype dot = caffe_cpu_dot(count, d_.cpu_data(), d_.cpu_data());
			//part2 of Loss
			Dtype d_sum = Dtype(0);
			Dtype* d_data = d_.mutable_cpu_data();
			for (int i = 0; i < count; i++) {
				d_sum += d_data[i];
			}

			LOG(INFO) << "dot: " << dot << "d_sum: " << d_sum;
			//double gamma = 0.5;
			Dtype loss = dot / count - gamma * d_sum * d_sum / count / count;

			top[0]->mutable_cpu_data()[0] = loss;

			//free memory space
			delete bottom1_log;
	}

	template <typename Dtype>
	void DepthLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
			if (propagate_down[1]) {
				LOG(FATAL) << this->type()
					<< " Layer cannot backpropagate to depth inputs.";
			}
			if (propagate_down[0]) {
				int count = bottom[0]->count();
				LOG(INFO) << "hello kugou";
				Dtype d_sum = Dtype(0);
				Dtype* d_data = d_.mutable_cpu_data();
				for (int i = 0; i < count; i++) {
					d_sum += d_data[i];
				}

				Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
				for (int i = 0; i < count; i++) {
					bottom_diff[i] = Dtype(2) * d_data[i] / count - Dtype(2) * gamma * d_sum / count / count;
				}
			}
	}

//#ifdef CPU_ONLY
//STUB_GPU(DepthLossLayer);
//#endif

INSTANTIATE_CLASS(DepthLossLayer);
REGISTER_LAYER_CLASS(DepthLoss);

} // namespace caffe