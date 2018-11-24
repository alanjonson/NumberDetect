# NumberDetect
 使用TensorFlow object detection API 训练了ssd+mobilenetV2，最后导出tflite模型，并且成功在ios上运行了 
 数据集使用的是 SVHN http://ufldl.stanford.edu/housenumbers/ 
 教程主要参照了谷歌官方 https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193 

#### 因为在win10，mac，Ubuntu上各跑了一遍，踩坑无数，事后记录一下踩过的坑（Linux是最好的操作系统！)

## Win10环境
### 版本  
    tensorflow-gpu 1.12.0
    protoc 3.0.0 or 3.3.0
### 数据集的制作
object detection API需要制作成tfrecord格式的数据集，SVHN的数据集很多标注的很乱，有的box远远超出图片，因此在转换格式的时候判断了一下框的位置，超过1（其实ssd最大允许1.1）的时候强制写回1
因此会出现如下样式的代码

    if (bbox['left'][i] + bbox['width'][i]) / width > 1:
        xmaxs.append(1)
        # print('xmaxs : %s' % (xmaxs))
    else:
        xmaxs.append((bbox['left'][i] + bbox['width'][i]) / width)

### checkpoint下载
因为采用了MobileNetV2，已经训练好的模型在 
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz 
下载,github上的2个模型都没跑起来

### cocoapi安装
win10上cocoapi需要安装visual studio 2015，最好全勾上，主要是要VC的platform SDK
要先 

    pip install Cython
    pip install pycocotools

### 模型导出
ssd模型需要用 [export_tflite_ssd_graph.py]() 导出，然后通过toco导出成tflite格式

## Mac环境

### 版本  
    osx 10.14.1
    tensorflow 1.12.0
    protoc 3.3.0
直接用pypi装是tensorflow是1.11的[（清华源）](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) ，得下载1.12。0的离线包.whl安装
记得先装用离线包装一个最新的tensorboard，mac没有gpu版本就不用装-gpu了

### Python环境
mac自带一个python2.7，装完python3.6.7以后建议在.bash_profile中加上（用venv环境最好）：

    PATH="/Library/Frameworks/Python.framework/Versions/3.6/bin:${PATH}"
    export PATH
    alias python="/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6"

## Ubuntu环境

### 版本  
    Ubuntu 16.04
    tensorflow 1.12.0
    protoc 3.3.0
直接用pypi装是tensorflow是1.11的（清华源），得下载1.12。0的离线包.whl安装
记得先装用离线包装一个最新的tensorboard

### Python环境
ubuntu也自带了python2.7，和python3.5，建议更新到3.6
用venv环境最好
经常会出现No module named "object_detection" 的报错，需要到tensorflow/models/research/ 目录中输入： 

    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

### pillow报错
开始训练的时候pillow报错了，发现Ubuntu自带的pillow比较老，要更新到最新版本（5.3.0）

### 训练config配置
如果要减小模型体积可以使用量化训练,int8位，比float的32位小了 3/4 但量化训练貌似更吃配置
float的时候batch_size = 16，GTX 960每步训练0.5 ~ 0.6秒 
[quantization]()模式下训练的时候batch_size =8 ，每步训练要1.2秒
这配置吃的也太高了，各个参数的作用还没不理解，具体原因待查

### 模型导出
转换成tflite格式的模型一定要使用bazel不然会报错：

    bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
    --input_file=$OUTPUT_DIR/tflite_graph.pb \
    --output_file=$OUTPUT_DIR/detect.tflite \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_values=128 \
    --change_concat_input_ranges=false \
    --allow_custom_ops
