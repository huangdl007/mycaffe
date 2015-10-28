#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void BBFaceLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		diff_.ReshapeLike(*bottom[0]);
		d_.ReshapeLike(*bottom[0]);
		H0_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	Dtype BBFaceLossLayer<Dtype>::computeDistance(int x, int y, int partK_x, int partK_y)
	{
		int stride = 5;
		int width = 64;
		int receptiveX = stride * x;
		int receptiveY = stride * y;

		//part k out of the receptive field
		if( partK_x < receptiveX || partK_x > (receptiveX+width) || 
			partK_y < receptiveY || partK_y > (receptiveY+width) )
		{
			return Dtype(-1);
		}

		int center_x = receptiveX + width/2;
		int center_y = receptiveY + width/2;

		return sqrt( (partK_x-center_x)*(partK_x-center_x) + 
					(partK_y-center_y)*(partK_y-center_y));
	}

	template <typename Dtype>
	void BBFaceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			int count = bottom[0]->count();
			int num = bottom[0]->num();
			CHECK_EQ(bottom[0]->count(), H0_.count())
				<< "Response map must have the same length";
			CHECK_EQ(d_.count(), H0_.count())
				<< "Response map must have the same length";
			CHECK_EQ(H0_.count(), num*2000)
				<< "Response length is not right";

			Dtype beta = Dtype(-0.1);

			Dtype* H0Data = H0_.mutable_cpu_data();
			Dtype* landmarkData = bottom[1]->mutable_cpu_data();
			//compute ground truth response map
			for(int slice = 0; slice < num; slice++)
			{
				int hasPartK = landmarkData[slice*11];
				for(int i = 0; i < 5; i++)
				{
					//coordinates x, y for part k
					int partK_x = landmarkData[slice*11 + 1 + i*2];
					int partK_y = landmarkData[slice*11 + 1 + i*2 + 1];
					//compute each part separately
					for(int j = 0; j < 20; j++)
					{
						for(int k = 0; k < 20; k++)
						{
							if(hasPartK == 0)
							{
								H0Data[slice*2000 + i*400 + j*20 + k] = Dtype(0);
							}
							else
							{
								//compute distance between part k and patch center
								Dtype r = computeDistance(j, k, partK_x, partK_y);
								H0Data[slice*2000 + i*400 + j*20 + k] = exp(beta*r);
							}
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
	void BBFaceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
			if (propagate_down[1]) {
				LOG(FATAL) << this->type()
					<< " Layer cannot backpropagate to landmarks inputs.";
			}
			if (propagate_down[0]) {
				int count = bottom[0]->count();

				Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
				Dtype* d_data = d_.mutable_cpu_data();
				for (int i = 0; i < count; i++) {
					bottom_diff[i] = d_data[i] / loss_;
				}
			}
	}

//#ifdef CPU_ONLY
//STUB_GPU(DepthLossLayer);
//#endif

INSTANTIATE_CLASS(BBFaceLossLayer);
REGISTER_LAYER_CLASS(BBFaceLoss);

} // namespace caffe