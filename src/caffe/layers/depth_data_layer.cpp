#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
	template <typename Dtype>
	DepthDataLayer<Dtype>::~DepthDataLayer<Dtype>() {
		this->JoinPrefetchThread();
	}

	template <typename Dtype>
	void DepthDataLayer<Dtype>::ReadDepthToArray(const string& filename, float* depths){
		std::ifstream infile(filename.c_str());
		float dep;
		int index = 0;
		while(infile >> dep) {
			depths[index++] = dep;
		}

	}

	template <typename Dtype>
	void DepthDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const int new_height = this->layer_param_.depth_data_param().new_height();
		const int new_width  = this->layer_param_.depth_data_param().new_width();
		const bool is_color  = this->layer_param_.depth_data_param().is_color();
		string root_folder   = this->layer_param_.depth_data_param().root_folder();

		CHECK((new_height == 0 && new_width == 0) ||
			(new_height > 0 && new_width > 0)) << "Current implementation requires "
			"new_height and new_width to be set at the same time.";
		// Read the file with image filenames and depth filenames
		const string& source = this->layer_param_.depth_data_param().source();
		LOG(INFO) << "Opening file " << source;
		std::ifstream infile(source.c_str());
		string image_filename;
		string depth_filename;
		while (infile >> image_filename >> depth_filename) {
			lines_.push_back(std::make_pair(image_filename, depth_filename));
		}
		LOG(INFO) << "A total of " << lines_.size() << " images.";

		lines_id_ = 0;

		//image
		// Read an image, and use it to initialize the top blob.
		cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
		// Use data_transformer to infer the expected blob shape from a cv_image.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
		this->transformed_data_.Reshape(top_shape);
		// Reshape prefetch_data and top[0] according to the batch_size.
		const int batch_size = this->layer_param_.depth_data_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";
		top_shape[0] = batch_size;
		this->prefetch_data_.Reshape(top_shape);
		top[0]->ReshapeLike(this->prefetch_data_);

		LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();

		//depths
		vector<int> label_shape;
		label_shape.push_back(batch_size);
		label_shape.push_back(74*74);
		top[1]->Reshape(label_shape);
		this->prefetch_label_.Reshape(label_shape);

		LOG(INFO) << "output depth size: " << label_shape[0] << ","
			<< label_shape[1];
		
	}

	// This function is used to create a thread that prefetches the data.
	template <typename Dtype>
	void DepthDataLayer<Dtype>::InternalThreadEntry() {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(this->prefetch_data_.count());
		CHECK(this->transformed_data_.count());
		//CHECK(this->transformed_data_.count());
		DepthDataParameter depth_data_param = this->layer_param_.depth_data_param();
		const int batch_size = depth_data_param.batch_size();
		const int new_height = depth_data_param.new_height();
		const int new_width = depth_data_param.new_width();
		const bool is_color = depth_data_param.is_color();
		string root_folder = depth_data_param.root_folder();

		// Reshape according to the first image of each batch
		// on single input batches allows for inputs of varying dimension.
		cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
			new_height, new_width, is_color);
		CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
		// Use data_transformer to infer the expected blob shape from a cv_img.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  		this->transformed_data_.Reshape(top_shape);
		// Reshape prefetch_data according to the batch_size.
		top_shape[0] = batch_size;
		this->prefetch_data_.Reshape(top_shape);

		Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
		Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();

		// datum scales
		const int lines_size = lines_.size();
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			// get a blob
			timer.Start();
			CHECK_GT(lines_size, lines_id_);
			cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
				new_height, new_width, is_color);
			CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
			read_time += timer.MicroSeconds();
			timer.Start();
			// Apply transformations (mirror, crop...) to the image
			int offset = this->prefetch_data_.offset(item_id);
			this->transformed_data_.set_cpu_data(prefetch_data + offset);
			this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
			trans_time += timer.MicroSeconds();

			//read Depths
			//prefetch_label[item_id] = lines_[lines_id_].second;
			float depths[74*74];
			ReadDepthToArray(lines_[lines_id_].second, depths);
			int depth_offset = this->prefetch_label_.offset(item_id);
			memcpy(&prefetch_label[depth_offset], &depths[0], sizeof(depths));

			// go to the next iter
			lines_id_++;
			if (lines_id_ >= lines_size) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				lines_id_ = 0;
			}
		}
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	INSTANTIATE_CLASS(DepthDataLayer);
	REGISTER_LAYER_CLASS(DepthData);
}	//namespace caffe