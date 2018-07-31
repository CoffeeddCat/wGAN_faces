import tensorflow as tf

def session(graph=None, allow_soft_placement=True,
            log_device_placement=False, allow_growth=True):
    """ return a Session with simple config """

    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                            log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)

def disk_image_batch(image_paths, batch_size, shape, preprocess_fn=None, shuffle=True, num_threads=16,
                     min_after_dequeue=100, allow_smaller_final_batch=False, scope=None):

    with tf.name_scope(scope, 'disk_image_batch'):
        data_num = len(image_paths)
        print(data_num)
        # dequeue a single image path and read the image bytes; enqueue the whole file list
        _, img = tf.WholeFileReader().read(tf.train.string_input_producer(image_paths, shuffle=shuffle, capacity=data_num))
        img = tf.image.decode_image(img)

        # preprocessing
        img.set_shape(shape)
        if preprocess_fn is not None:
            img = preprocess_fn(img)

        # batch datas
        if shuffle:
            capacity = min_after_dequeue + (num_threads + 1) * batch_size
            img_batch = tf.train.shuffle_batch([img],
                                               batch_size=batch_size,
                                               capacity=capacity,
                                               min_after_dequeue=min_after_dequeue,
                                               num_threads=num_threads,
                                               allow_smaller_final_batch=allow_smaller_final_batch)
        else:
            img_batch = tf.train.batch([img],
                                       batch_size=batch_size,
                                       allow_smaller_final_batch=allow_smaller_final_batch)

        return img_batch, data_num


class dataset:

    def __init__(self, image_paths, batch_size, shape, preprocess_fn=None, shuffle=True, num_threads=16,
                 min_after_dequeue=100, allow_smaller_final_batch=False, scope=None):

        self.graph = tf.Graph()  # declare ops in a separated graph
        with self.graph.as_default():
            # @TODO
            # There are some strange errors if the gpu device is the
            # same with the main graph, but cpu device is ok. I don't know why...
            with tf.device('/cpu:0'):
                self._batch_ops, self._data_num = disk_image_batch(image_paths, batch_size, shape, preprocess_fn, shuffle, num_threads,
                                                                   min_after_dequeue, allow_smaller_final_batch, scope)

        print(' [*] DiskImageData: create session!')
        self.sess = session(graph=self.graph)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def __len__(self):
        return self._data_num

    def batch(self):
        return self.sess.run(self._batch_ops)

    def __del__(self):
        print(' [*] DiskImageData: stop threads and close session!')
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()