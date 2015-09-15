caffe_root = '/home/huangdl007/caffe/'
import sys
sys.path.insert(0, caffe_root+'python')
import caffe
import numpy as np
import cv2

deploy_ptototxt = 'local_deploy.prototxt'
model = 'NYU_DEPTH_LOCAL_iter_200000.caffemodel'

#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(deploy_ptototxt, model, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
net.blobs['data'].reshape(32,3,304,304)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('test.png'))
out = net.forward()


predicted_depth = out['f-conv4'][0][0]
print predicted_depth
for i in np.arange(len(predicted_depth)):
	for j in np.arange(len(predicted_depth[0])):
		predicted_depth[i][j] = np.exp(predicted_depth[i][j])

max_depth = np.max(predicted_depth)
print max_depth
img = np.zeros((74,74), np.uint8)
for i in np.arange(len(predicted_depth)):
	for j in np.arange(len(predicted_depth[0])):
		img[i][j] = np.ceil(predicted_depth[i][j]/max_depth*255)


print img
cv2.namedWindow('fuck')
cv2.imshow('fuck', img)
cv2.waitKey(0)
#cv2.imwrite('result_300000.jpg', img)
cv2.destroyAllWindows()