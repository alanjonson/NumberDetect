import tensorflow as tf
from PIL import Image

slim = tf.contrib.slim


def get_record_dataset(record_path,
                       reader=tf.TFRecordReader, image_shape=[-1, -1],
                       num_samples=10, num_classes=10):
    keys_to_features = {
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value='1'),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value='1'),
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value='1'),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),

        # 'image/object/class/text':
        #     tf.VarLenFeature([], tf.string),
        # 'image/object/class/label':
        #     tf.VarLenFeature([], tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                              format_key='image/format',
                                              channels=3),
        'image/width': slim.tfexample_decoder.Tensor('image/width'),
        'image/height': slim.tfexample_decoder.Tensor('image/height'),
        'image/filename': slim.tfexample_decoder.Tensor('image/filename'),
        # 'image/object/class/text': slim.tfexample_decoder.Tensor('image/object/class/text'),
        'image/object/class/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
        # 'image/object/bbox/xmin': slim.tfexample_decoder.Tensor('image/object/bbox/xmin'),
        # 'image/object/bbox/xmax': slim.tfexample_decoder.Tensor('image/object/bbox/xmax'),
        # 'image/object/bbox/ymin': slim.tfexample_decoder.Tensor('image/object/bbox/ymin'),
        # 'image/object/bbox/ymax': slim.tfexample_decoder.Tensor('image/object/bbox/ymax')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    items_to_descriptions = {
        'image': 'image',
        'label': 'label'}

    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)


if __name__ == '__main__':
    dataSet = get_record_dataset(record_path='tfrecord_output.record', num_classes=10)
    print(dataSet)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataSet, num_readers=1, common_queue_capacity=20,
                                                              common_queue_min=1)

    [image, height, width] = provider.get(['image', 'image/height', 'image/width'])
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(2):
            img, h, w = sess.run([image, height, width])
            img = tf.reshape(img, [h, w, 3])
            print(img.shape)
            img = Image.fromarray(img.eval(), 'RGB')  # 这里将narray转为Image类，Image转narray：a=np.array(img)
            # img.save('./' + str(l) + '.jpg')  # 保存图片
            img.show()

        coord.request_stop()
        coord.join(threads)
