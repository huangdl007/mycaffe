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
	void BBFaceLossLayer<Dtype>::GetReceptiveField(vector<int>& region, const int kernel_size, const int stride,
		const int pad, const int pool_size, const int max_x, const int max_y)
	{
		// Old region
		int x0 = region[0] * pool_size;
		int y0 = region[1] * pool_size;
		int x1 = region[2] * pool_size;
		int y1 = region[3] * pool_size;

		// Compute right bottom first
		region[2] = kernel_size + (x1 - 1)*stride - pad;
		region[3] = kernel_size + (y1 - 1)*stride - pad;

		// Compute left top according to the right bottom
		region[0] = region[2] - (x1 - x0 - 1)*stride - kernel_size;
		region[1] = region[3] - (y1 - y0 - 1)*stride - kernel_size;
		region[0] = region[0] > 0 ? region[0] : 0;
		region[1] = region[1] > 0 ? region[1] : 0;

		// Right bottom need to be LE max
		region[2] = region[2] > max_x ? region[2] - pad : region[2];
		region[3] = region[3] > max_y ? region[3] - pad : region[3];
	}

	template <typename Dtype>
	Dtype BBFaceLossLayer<Dtype>::computeDistance(const int x, const int y, const int partK_x, const int partK_y)
	{
		// Build the original region
		vector<int> region(4);
		region[0] = x;
		region[1] = y;
		region[2] = x+1;
		region[3] = y+1;

		// Get the receptive feild layer by layer from bottom to top
		GetReceptiveField(region, 9, 1, 4, 1, 20, 20);
		GetReceptiveField(region, 5, 1, 2, 2, 40, 40);
		GetReceptiveField(region, 5, 1, 2, 2, 80, 80);
		GetReceptiveField(region, 5, 1, 2, 2, 160, 160);
		//part k out of the receptive field
		if( partK_x < region[0] || partK_x > region[2] || 
			partK_y < region[1] || partK_y > region[3] )
		{
			return Dtype(-1);
		}

		int center_x = (region[0] + region[2]) / 2;
		int center_y = (region[1] + region[3]) / 2;

		return Dtype(sqrt( (partK_x-center_x)*(partK_x-center_x) + 
					(partK_y-center_y)*(partK_y-center_y)));
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
				for(int part = 0; part < 5; part++)
				{
					//coordinates x, y for part k
					int partK_x = landmarkData[slice*11 + 1 + part*2];
					int partK_y = landmarkData[slice*11 + 1 + part*2 + 1];
					//compute each part separately
					for(int h = 0; h < 20; h++)
					{
						for(int w = 0; w < 20; w++)
						{
							if(hasPartK == 0)
							{
								H0Data[slice*2000 + part*400 + h*20 + w] = Dtype(0);
							}
							else
							{
								//compute distance between part k and patch center
								Dtype r = computeDistance(w, h, partK_x, partK_y);
								if(r == Dtype(-1))
								{
									H0Data[slice*2000 + part*400 + h*20 + w] = Dtype(0);
								}
								else
								{
									H0Data[slice*2000 + part*400 + h*20 + w] = exp(beta*r);
								}
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