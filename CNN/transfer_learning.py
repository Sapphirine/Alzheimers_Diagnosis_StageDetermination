# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright (C) 2017 Wenda Zhou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Utility code for the fine-tuning homework.

This file contains convenience functions to facilitate
fine-tuning an inception model. If you wish to understand
all the details, you may wish to read through the file.

If the name of the function starts with and underscore,
it is less important.
"""

import os
import re
import sys
import urllib, urllib.request
import tarfile
import tensorflow as tf
import numpy as np
import typing


def create_model():
    """Loads and creates a graph from the saved model.

    Returns
    -------
    graph: A tensorflow graph representing the loaded graph
    bottleneck: A tensor in the graph representing the bottleneck layer
    resized_input: A tensor in the graph representing the input layer
    softmax: A tensor in the graph representing the softmax layer
    """
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    _maybe_download_and_extract(data_url)

    graph, bottleneck, resized_input, softmax = _create_model_graph()

    return graph, bottleneck, resized_input, softmax


_MODEL_DIR = './model'


def _maybe_download_and_extract(data_url):
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.

    Args:
      data_url: Web location of the tar file containing the pretrained model.
    """
    dest_directory = _MODEL_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
                        'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _create_model_graph() -> typing.Tuple[tf.Graph, tf.Tensor, tf.Tensor, tf.Tensor]:
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    This creates the graph for the Inception model that was downloaded.

    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(_MODEL_DIR, 'classify_image_graph_def.pb')
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck, resized_input, softmax = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    'pool_3/_reshape:0',  # bottleneck tensor name
                    'Mul:0',  # resized input tensor name
                    'softmax:0',  # predicted probability tensor name
                ]))
    return graph, bottleneck, resized_input, softmax


def make_jpeg_decoding() -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """Adds operations that perform JPEG decoding and resizing to the graph.

    Returns
    -------
    jpeg_data: The tensor to feed the jpeg data.
    mul_image: The tensor representing the image.
    """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=3)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([299, 299])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, 128)
    mul_image = tf.multiply(offset_image, 1.0 / 128)
    return jpeg_data, mul_image


def _get_bottleneck_path(image_path, bottleneck_dir):
    image_name = os.path.basename(image_path)
    bottleneck_path = os.path.join(bottleneck_dir, image_name + '.npy')
    return bottleneck_path


def _cache_single_bottleneck(compute_bottleneck, image_path, bottleneck_dir):
    bottleneck_path = _get_bottleneck_path(image_path, bottleneck_dir)

    if os.path.isfile(bottleneck_path):
        return False

    with open(image_path, 'rb') as f:
        jpeg_data = f.read()

    bottleneck = compute_bottleneck(jpeg_data)
    bottleneck = np.squeeze(bottleneck)
    np.save(bottleneck_path, bottleneck, allow_pickle=False)
    return True


def _load_single_bottleneck(image_path, bottleneck_dir):
    bottleneck_path = _get_bottleneck_path(image_path, bottleneck_dir)

    return np.load(bottleneck_path, allow_pickle=False)


def cache_bottlenecks(compute_bottleneck, session, image_list):
    """Computes and saves the bottlenecks for each image in the list

    This saves all the the bottlenecks into the ./data/bottlenecks folder.
    For each image, we check if the bottleneck exists, and if not, we
    compute it. If you want to regenerate the bottlenecks, you will need
    to delete the files.

    Parameters
    ----------
    compute_bottleneck: The function to compute the bottleneck.
    session: The tensorflow session to execute the operations.
    jpeg_data_tensor: The placeholder for the jpeg input data.
    bottleneck_tensor: The tensor representing the bottleneck
    image_list: A list of images for which to compute the bottlenecks
    """
    bottlenecks_dir = './data/bottlenecks'

    def do_compute_bottleneck(image_data):
        return compute_bottleneck(session, image_data)

    if not os.path.isdir(bottlenecks_dir):
        print('Creating directory for bottlenecks')
        os.makedirs(bottlenecks_dir)

    any_computed = False

    for i, (label, image) in enumerate(image_list):
        any_computed |= _cache_single_bottleneck(do_compute_bottleneck, image, bottlenecks_dir)

        if i % 20 == 0:
            print('Saved {0}/{1} bottlenecks'.format(i + 1, len(image_list)))
    else:
        print('Done computing bottlenecks!')

    if not any_computed:
        print('No bottlenecks were computed as they already exist!')


def create_training_dataset(image_list):
    """Creates a training dataset from the bottlenecks.

    Parameters
    ----------
    image_list: the list of images to load.

    Returns
    -------
    A dictionary with labels and bottlenecks.
    labels: a vector of labels.
    bottlenecks: a matrix of bottlenecks.
    """
    labels = []
    bottlenecks = []

    for label, image in image_list:
        labels.append(label)
        bottlenecks.append(_load_single_bottleneck(image, './data/bottlenecks'))

    return {
        'labels': np.stack(labels),
        'bottlenecks': np.stack(bottlenecks)
    }


def create_image_lists(image_dir, testing_percentage, max_number_images=200):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Parameters
    ----------
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    max_number_images: The maximum number of images to consider for every class.

    Returns
    -------
    A list of training examples, a list of testing examples, and a dictionary
    mapping indices to labels.
    """
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in tf.gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > max_number_images:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, max_number_images))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        for file_name in sorted(file_list)[:max_number_images]:
            base_name = os.path.basename(file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hashed = hash(base_name)
            percentage_hash = ((hashed %
                                (max_number_images + 1)) *
                               (100.0 / max_number_images))
            if percentage_hash < testing_percentage:
                testing_images.append(file_name)
            else:
                training_images.append(file_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
        }

    training = []
    testing = []

    label_map = {}
    label_idx = 0

    for label_name, data in result.items():
        label_map[label_idx] = label_name

        for image_path in data['training']:
            training.append((label_idx, image_path))

        for image_path in data['testing']:
            testing.append((label_idx, image_path))

        label_idx += 1

    return training, testing, label_map


def get_testing_data(image_list) -> typing.Tuple[typing.List[int], typing.List[bytes]]:
    """Gets the testing data.

    Parameters
    ----------
    image_list: the list of testing images to create a dataset.

    Returns
    -------
    labels: a vector of labels.
    images: a vector of image data.
    """
    labels = []
    images = []

    for label, path in image_list:
        labels.append(label)

        with tf.gfile.Open(path, 'rb') as f:
            images.append(f.read())

    return labels, images
