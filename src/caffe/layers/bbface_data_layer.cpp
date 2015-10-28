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
	BBFaceDataLayer<Dtype>::~BBFaceDataLayer<Dtype>() {
		this->JoinPrefetchThread();
	}

	template <typename Dtype>
	void BBFaceDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const int new_height = this->layer_param_.bbface_data_param().new_height();
		const int new_width  = this->layer_param_.bbface_data_param().new_width();
		const bool is_color  = this->layer_param_.bbface_data_param().is_color();
		const bool is_shuffled  = this->layer_param_.bbface_data_param().shuffle();
		string root_folder   = this->layer_param_.bbface_data_param().root_folder();

		CHECK((new_height == 0 && new_width == 0) ||
			(new_height > 0 && new_width > 0)) << "Current implementation requires "
			"new_height and new_width to be set at the same time.";
		// Read the file with image filenames and landmarks info
		const string& source = this->layer_param_.bbface_data_param().source();
		LOG(INFO) << "Opening file " << source;
		std::ifstream infile(source.c_str());
		string image_filename;
		while (infile >> image_filename) {
			vector<int> landmarks;
			int tmp;
			for(int i = 0; i < 11; i++)
			{
				infile >> tmp;
				landmarks.push_back(tmp);
			}
			lines_.push_back(std::make_pair(image_filename, landmarks));
		}
		infile.close();
		
		// randomly shuffle data
		if(is_shuffled)
		{
			LOG(INFO) << "Shuffleing data";
			const unsigned int prefetch_rng_seed = caffe_rng_rand();
			prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
			ShuffleImages();
			LOG(INFO) << "A total of " << lines_.size() << " images.";
		}

		lines_id_ = 0;

		//image
		// Read an image, and use it to initialize the top blob.
		cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
		// Use data_transformer to infer the expected blob shape from a cv_image.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
		this->transformed_data_.Reshape(top_shape);
		// Reshape prefetch_data and top[0] according to the batch_size.
		const int batch_size = this->layer_param_.bbface_data_param().batch_size();
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
		label_shape.push_back(11);
		top[1]->Reshape(label_shape);
		this->prefetch_label_.Reshape(label_shape);

		LOG(INFO) << "output landmark size: " << label_shape[0] << ","
			<< label_shape[1];
		
	}

	template <typename Dtype>
	void BBFaceDataLayer<Dtype>::ShuffleImages() {
	  caffe::rng_t* prefetch_rng =
	      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	}

	// This function is used to create a thread that prefetches the data.
	template <typename Dtype>
	void BBFaceDataLayer<Dtype>::InternalThreadEntry() {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(this->prefetch_data_.count());
		CHECK(this->transformed_data_.count());
		//CHECK(this->transformed_data_.count());
		BBFaceDataParameter bbface_data_param = this->layer_param_.bbface_data_param();
		const int batch_size = bbface_data_param.batch_size();
		const int new_height = bbface_data_param.new_height();
		const int new_width = bbface_data_param.new_width();
		const bool is_color = bbface_data_param.is_color();
		string root_folder = bbface_data_param.root_folder();
		const bool is_shuffled = bbface_data_param.shuffle();

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
			
			int offset = this->prefetch_data_.offset(item_id);
			this->transformed_data_.set_cpu_data(prefetch_data + offset);
			this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
			trans_time += timer.MicroSeconds();

			//read landmarks
			//prefetch_label[item_id] = lines_[lines_id_].second;
			for(int i = item_id*11, j = 0; i < (item_id+1)*11 && j < 11; i++, j++)
			{
				prefetch_label[i] = lines_[lines_id_].second[j];
			}

			// go to the next iter
			lines_id_++;
			if (lines_id_ >= lines_size) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				lines_id_ = 0;
				if(is_shuffled)
				{
					ShuffleImages();
				}
			}
		}
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	INSTANTIATE_CLASS(BBFaceDataLayer);
	REGISTER_LAYER_CLASS(BBFaceData);
}	//namespace caffe