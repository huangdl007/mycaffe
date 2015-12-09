#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void BHFaceLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		d_.ReshapeLike(*bottom[0]);
		H0_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	Dtype BHFaceLossLayer<Dtype>::ComputeDistance(const int x, const int y, const int partK_x, const int partK_y)
	{
		return Dtype(sqrt( (partK_x-x)*(partK_x-x) + 
					(partK_y-y)*(partK_y-y)));
	}

	template <typename Dtype>
	void BHFaceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			int count = bottom[0]->count();
			int num = bottom[0]->num();
			CHECK_EQ(bottom[0]->count(), H0_.count())
				<< "Response map must have the same length";
			CHECK_EQ(d_.count(), H0_.count())
				<< "Response map must have the same length";

			Dtype beta = Dtype(-0.15);

			Dtype* H0Data = H0_.mutable_cpu_data();
			const Dtype* landmarkData = bottom[1]->cpu_data();
			const Dtype* startXYData = bottom[2]->cpu_data();
			//compute ground truth response map
			const int height = bottom[0]->height();
			const int width = bottom[0]->width();
			const int length = height*width;
			for(int slice = 0; slice < num; slice++)
			{
				const int hasPartK = landmarkData[slice*11];
				const int part = startXYData[slice*3 + 0];
				//coordinates x, y for part k
				const int partK_x = landmarkData[slice*11 + 1 + part*2];
				const int partK_y = landmarkData[slice*11 + 1 + part*2 + 1];
				const int startX = startXYData[slice*3 + 1];
				const int startY = startXYData[slice*3 + 2];

				for(int h = 0; h < height; h++)
				{
					for(int w = 0; w < width; w++)
					{
						if(hasPartK == 0)
						{
							H0Data[slice*length + h*height + w] = Dtype(0);
						}
						else
						{
							//compute distance between part k and patch center
							Dtype r = ComputeDistance(startX+w, startY+h, partK_x, partK_y);
							H0Data[slice*length + h*height + w] = exp(beta*r);
						}
					}
				}
			}

			//compute H - H0
			caffe_sub(
				count,
				bottom[0]->cpu_data(),
				H0_.cpu_data(),
				d_.mutable_cpu_data());
			
			//compute loss
			Dtype dot = caffe_cpu_dot(
								count,
								d_.cpu_data(),
								d_.cpu_data());
			loss_ = sqrt(dot);
			top[0]->mutable_cpu_data()[0] = loss_;
	}

	template <typename Dtype>
	void BHFaceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
			if (propagate_down[1]) {
				LOG(FATAL) << this->type()
					<< " Layer cannot backpropagate to landmarks inputs.";
			}
			if (propagate_down[0]) {
				int count = bottom[0]->count();

				Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
				const Dtype* d_data = d_.cpu_data();
				for (int i = 0; i < count; i++) {
					bottom_diff[i] = d_data[i] / loss_;
				}
			}
	}

//#ifdef CPU_ONLY
//STUB_GPU(BBFaceLossLayer);
//#endif

INSTANTIATE_CLASS(BHFaceLossLayer);
REGISTER_LAYER_CLASS(BHFaceLoss);

} // namespace caffe