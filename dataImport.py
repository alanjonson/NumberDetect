from scipy.io import loadmat
import h5py
from PIL import Image
import tensorflow as tf
import os
import io
from collections import namedtuple

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

train_dir = 'train/'
test_dir = 'test/'
mat_name = 'digitStruct.mat'
# 分别生成训练集和测试集
trans_dir = train_dir
dataNum = 33000


class dataImport:
    def __init__(self, file):
        self.file = h5py.File(file, 'r')
        self.file_name = self.file['digitStruct']['name']
        self.file_bbox = self.file['digitStruct']['bbox']

    def get_img_name(self, n):
        name = ''.join([chr(c[0]) for c in self.file[self.file_name[n][0]].value])
        return name

    # 调试用
    def get_img_size(self, n):
        img = Image.open(trans_dir + self.get_img_name(n))
        print(img.size)

    def bbox_helper(self, attr):
        if len(attr) > 1:
            attr = [self.file[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

    def get_img_bbox(self, n):
        bbox = {}
        bb = self.file_bbox[n].item()
        bbox['label'] = self.bbox_helper(self.file[bb]["label"])
        bbox['top'] = self.bbox_helper(self.file[bb]["top"])
        bbox['left'] = self.bbox_helper(self.file[bb]["left"])
        bbox['height'] = self.bbox_helper(self.file[bb]["height"])
        bbox['width'] = self.bbox_helper(self.file[bb]["width"])
        return bbox

    def get_bbox_keys(self):
        for b in self.file[self.file_bbox[0].item()].keys():
            print(b)


def create_tf_example(filename, bbox, images_path):
    print(bbox)
    with tf.gfile.GFile(images_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    # print("name : %s width: %s height %s" % (filename, width, height))
    filename = filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for i in range(len(bbox['left'])):

        xmins.append(bbox['left'][i] / width)

        if (bbox['left'][i] + bbox['width'][i]) / width > 1:
            xmaxs.append(1)
            # print('xmaxs : %s' % (xmaxs))
        else:
            xmaxs.append((bbox['left'][i] + bbox['width'][i]) / width)
        ymins.append(bbox['top'][i] / height)

        if (bbox['top'][i] + bbox['height'][i]) / height > 1:
            ymaxs.append(1)
            # print('ymaxs : %s' % (ymaxs))
        else:
            ymaxs.append((bbox['top'][i] + bbox['height'][i]) / height)

        labelString = '%s' % (int(bbox['label'][i]))
        # if labelString == '10':
        #     labelString = '0'
        classes_text.append(labelString.encode('utf8'))
        classes.append(label_map_dict[labelString])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in
            zip(gb.groups.keys(), gb.groups)]


data = dataImport(trans_dir + mat_name)
# num = 0
# print(data.get_img_name(num))
# data.get_img_size(num)
# print(data.get_img_bbox(num))

flags = tf.app.flags
flags.DEFINE_string('label_map_path', 'tfrecord_output.pbtxt', 'Path to label map proto')

if trans_dir == train_dir:
    flags.DEFINE_string('output_path', 'number_train.record', 'Path to output TFRecord')
else:
    flags.DEFINE_string('output_path', 'number_test.record', 'Path to output TFRecord')

FLAGS = flags.FLAGS

writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

for i in range(0, dataNum):
    name = data.get_img_name(i)
    bbox = data.get_img_bbox(i)
    tf_example = create_tf_example(name, bbox, (trans_dir + name))
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = FLAGS.output_path
print('Successfully created the TFRecords: {}'.format(output_path))
