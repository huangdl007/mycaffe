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
	}

	template <typename Dtype>
	void DepthLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			int count = bottom[0]->count();
			Dtype* bottom0_log = new Dtype[count];
			Dtype* bottom1_log = new Dtype[count];

			//compute log for Y and Y*
			for (int i = 0; i < count; i++)
			{
				//predicted depth maybe < 0
				bottom0_log[i] = log(abs(bottom[0]->cpu_data()[i])+1);
				bottom1_log[i] = log(bottom[1]->cpu_data()[i]);
				
				if(i < 5)
				{
					LOG(INFO) << "pre depth: " << bottom[0]->cpu_data()[i] <<  ", log: " << bottom0_log[i];
					//LOG(INFO) << "tru depth: " << bottom[1]->cpu_data()[i] <<  ", log: " << bottom1_log[i];
				}
				
			}

			//compute di = logY - logY*
			caffe_sub(
				count,
				bottom0_log,
				bottom1_log,
				diff_.mutable_cpu_data());

			//part1 of Loss
			Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
			//part2 of Loss
			Dtype log_sum = Dtype(0);
			Dtype* diff_data = diff_.mutable_cpu_data();
			for (int i = 0; i < count; i++) {
				log_sum += diff_data[i];
			}

			//LOG(INFO) << "Dot: " << dot << ", Num: " << bottom[0]->num() << ", log_sum " << log_sum << ", gamma: " << gamma;
			//double gamma = 0.5;
			Dtype loss = dot / count - gamma * log_sum * log_sum / count / count;

			top[0]->mutable_cpu_data()[0] = loss;

			//free memory space
			delete bottom0_log;
			delete bottom1_log;
	}

	template <typename Dtype>
	void DepthLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
			if (propagate_down[1]) {
				LOG(FATAL) << this->type()
					<< " Layer cannot backpropagate to label inputs.";
			}
			if (propagate_down[0]) {
				int count = bottom[0]->count();

				Dtype log_sum = Dtype(0);
				Dtype* diff_data = diff_.mutable_cpu_data();
				for (int i = 0; i < count; i++) {
					log_sum += diff_data[i];
				}

				Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
				Dtype* bottom_data = bottom[0]->mutable_cpu_data();
				for (int i = 0; i < count; i++) {
					bottom_diff[i] = Dtype(2) * diff_data[i] / count / bottom_data[i] - Dtype(2) * gamma * log_sum / count / count / diff_data[i];

					if(i<5)
					{
						LOG(INFO) << "bottom_diff: " << bottom_diff[i];
					}
				}
				//ta da
			}
	}

//#ifdef CPU_ONLY
//STUB_GPU(DepthsLossLayer);
//#endif

INSTANTIATE_CLASS(DepthLossLayer);
REGISTER_LAYER_CLASS(DepthLoss);

} // namespace caffe