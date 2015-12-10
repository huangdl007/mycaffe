#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
const int BHFaceCropLayer<Dtype>::crop_height = 64;
template <typename Dtype>
const int BHFaceCropLayer<Dtype>::crop_width = 64;

template <typename Dtype>
void BHFaceCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// Bottom layer shape
	vector<int> shape = bottom[0]->shape();
	// Crop patch window size

	shape[2] = crop_height;
	shape[3] = crop_width;

	top[0]->Reshape(shape);
}

template <typename Dtype>
void BHFaceCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// Bottom layer shape
	vector<int> shape = bottom[0]->shape();
	// Crop patch window size

	shape[2] = crop_height;
	shape[3] = crop_width;

	top[0]->Reshape(shape);	
}

template <typename Dtype>
void BHFaceCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* window_size = bottom[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	for(int n = 0; n < top[0]->num(); ++n)
	{	
		const int start_h = (int)window_size[n*3 + 1];
		const int start_w = (int)window_size[n*3 + 2];
		CHECK(start_h + top[0]->height() <= bottom[0]->height() &&
				start_w + top[0]->width() <= bottom[0]->width())
		 	<< "Crop patch must inside the previous layer.";
		for(int c = 0; c < top[0]->channels(); ++c)
		{
			for(int h = 0; h < top[0]->height(); ++h)
			{
				caffe_copy(top[0]->width(),
					bottom_data + bottom[0]->offset(n, c, start_h+h, start_w),
					top_data + top[0]->offset(n, c, h));
			}
		}
	}
}

template <typename Dtype>
void BHFaceCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	const Dtype* window_size = bottom[1]->cpu_data();
	if(propagate_down[0])
	{
		caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);

		for(int n = 0; n < top[0]->num(); ++n)
		{
			const int start_h = (int)window_size[n*3 + 1];
			const int start_w = (int)window_size[n*3 + 2];
			for(int c = 0; c < top[0]->channels(); ++c)
			{
				for(int h = 0; h < top[0]->height(); ++h)
				{
					caffe_copy(top[0]->width(),
						top_diff + top[0]->offset(n, c, h),
						bottom_diff + bottom[0]->offset(n, c, start_h+h, start_w));
				}
			}
		}
	}
}
//#ifdef CPU_ONLY
//STUB_GPU(BHFaceCroplayer);
//#endif

INSTANTIATE_CLASS(BHFaceCropLayer);
REGISTER_LAYER_CLASS(BHFaceCrop);
}	//namespace caffe