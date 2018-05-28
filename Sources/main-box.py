import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image as ImageKeras
from keras.applications.resnet50 import preprocess_input
import os
import sys
import tensorflow as tf
import collections
from matplotlib import pyplot as plt
from PIL import Image
import math
import json
import time

start = time.time()

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..\\Object Detection Google\\models\\research")

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

# What model to download.
MODEL_NAME = '..\\Object Detection Google\\models\\research\\faster_rcnn_resnet101_lowproposals_coco_2018_01_28'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('..\\Object Detection Google\\models\\research\\object_detection\\data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def cosine_similarity(first, second):
    # (first . second)/{||first||*||second||)
    fs, ff, ss = 0, 0, 0
    for i in range(len(first)):
        f = first[i]
        s = second[i]
        ff += f*f
        ss += s*s
        fs += f*s
    return fs/math.sqrt(ff*ss)


# load model just once to use it in feature_extract
model = ResNet50(weights='imagenet', include_top=False)


def feature_extract(current_image):
    input_shape = (224, 224)
    # curr_image = load_image_into_numpy_array(current_image)
    # fig = plt.figure(figsize=(12, 8))
    # plt.imshow(curr_image)
    # plt.show()
    img = current_image.resize(input_shape, Image.NEAREST)
    # img = ImageKeras.load_img('temp.jpg', target_size=inputShape)
    x = ImageKeras.img_to_array(img)
    # our input image is now represented as a NumPy array of shape
    # (inputShape[0], inputShape[1], 3) however we need to expand the
    # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features_np = np.array(features).flatten()

    return features_np


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)


            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict


to_add_json = collections.defaultdict(list)


def run_inference_for_database_images_and_write_json(graph, files, json_info):
    with graph.as_default():
        with tf.Session() as sess:
            # Run inference and modify info json
            for image_path in files:
                if (image_path not in json_info) or (json_info[image_path]['date'] != os.path.getmtime(image_path)):

                    image = Image.open(image_path)
                    image_np = load_image_into_numpy_array(image)
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    # Get handles to input and output tensors
                    ops = tf.get_default_graph().get_operations()
                    all_tensor_names = {output.name for op in ops for output in op.outputs}
                    tensor_dict = {}
                    for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes'
                    ]:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                    if 'detection_masks' in tensor_dict:
                        # The following processing is only for single image
                        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                            detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                        detection_masks_reframed = tf.cast(
                            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                        tensor_dict['detection_masks'] = tf.expand_dims(
                            detection_masks_reframed, 0)

                    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                    # Actual detection
                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]


                    
                    # Get each box and label of the results of detection
                    cropped_images = get_boxes_and_labels_on_image_array(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index)

                    # vis_util.visualize_boxes_and_labels_on_image_array(
                    #     image_np,
                    #     output_dict['detection_boxes'],
                    #     output_dict['detection_classes'],
                    #     output_dict['detection_scores'],
                    #     category_index,
                    #     instance_masks=output_dict.get('detection_masks'),
                    #     use_normalized_coordinates=True,
                    #     line_thickness=8)

                    # fig = plt.figure(figsize=(12, 8))
                    # plt.imshow(image_np)
                    # plt.show()

                    if image_path in json_info:
                        del json_info[image_path]
                    json_info[image_path] = {}
                    json_info[image_path]['date'] = os.path.getmtime(image_path)
                    json_info[image_path]['crop'] = {}
                    to_add_json[image_path] = cropped_images


# Extract features of the results of detection and add them to json
def complete_json_with_feature_vectors(json_info):
    for image_path, cropped_images in to_add_json.items():
        for cropped_image in cropped_images:
            feature = feature_extract(cropped_image[1])
            cropped_class = cropped_image[0]
            if cropped_class not in json_info[image_path]['crop']:
                json_info[image_path]['crop'][cropped_class] = []
            json_info[image_path]['crop'][cropped_class].append(feature.tolist())


def get_mask_on_image_array(image, mask):
    [im_height, im_width, _] = image.shape
    for i in range(im_height):
        for j in range(im_width):
            if mask[i, j] == 0:
                image[i, j] = (255, 255, 255)
    return image


def get_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        max_boxes_to_draw=20,
        min_score_threshold=.5):
    # dictionary box [y1,y2,x1,x2] -> info string ['label:score']
    box_to_info_str_map = collections.defaultdict(list)
    box_to_instance_masks_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_threshold:
            box = tuple(boxes[i].tolist())

            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]

            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'N/A'
            info_str = str(class_name)
            box_to_info_str_map[box].append(info_str)

    cropped_images = []

    for box, info_str in box_to_info_str_map.items():
        ymin, xmin, ymax, xmax = box
        [im_height, im_width, _] = image.shape
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        crop_image = np.copy(image[int(top):int(bottom), int(left):int(right), : ])

        image_pil = Image.fromarray(np.uint8(crop_image)).convert('RGB')
        cropped_images.append([''.join(info_str), image_pil])


    return cropped_images


# Create database of indexed images
startdb = time.time()
JSON_PATH = 'infobox.json'
files = []
for (path, dirnames, filenames) in os.walk(os.getcwd() + '\\images'):
    files.extend(os.path.join(path, name) for name in filenames)

with open(JSON_PATH, 'r') as fin:
    json_info = json.load(fin)

run_inference_for_database_images_and_write_json(detection_graph, files, json_info)
complete_json_with_feature_vectors(json_info)

with open(JSON_PATH, 'w') as fout:
    json.dump(json_info, fout)
enddb = time.time()

# Test one image
currentImagePath = 'test_image.jpg'
image = Image.open(currentImagePath)
image_np = load_image_into_numpy_array(image)
image_np_expanded = np.expand_dims(image_np, axis=0)
# Actual detection.
output_dict = run_inference_for_single_image(image_np, detection_graph)

# Get each box and label of the results of detection
cropped_images = get_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks'))

# fig = plt.figure(figsize=(12, 8))
# plt.imshow(image_np)
# plt.show()

im_copy = np.copy(image_np)
vis_util.visualize_boxes_and_labels_on_image_array(
    im_copy,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks'),
    use_normalized_coordinates=True,
    line_thickness=4)

fig = plt.figure(figsize=(12, 8))
plt.imshow(im_copy)
plt.axis('off')
plt.title('Faster R-CNN Object Detection Result')
plt.show()


# Find images in database which contains similar objects
to_add_json_compare = collections.defaultdict(list)
find = time.time()
for cropped_image in cropped_images:
    feature = feature_extract(cropped_image[1])
    cropped_class = cropped_image[0]
    correct_images = []
    all_images = {}
    for compareImage_path in files:
        if compareImage_path in json_info:
            if cropped_class in json_info[compareImage_path]['crop']:
                for compareFeature in json_info[compareImage_path]['crop'][cropped_class]:
                    if compareImage_path not in correct_images:
                        result = cosine_similarity(feature, compareFeature)
                        print(result)
                        if compareImage_path in all_images and all_images[compareImage_path] < result:
                            all_images[compareImage_path] = result
                        else:
                            all_images[compareImage_path] = result
                        if result > 0.57:
                            correct_images.append(compareImage_path)
                            for c in json_info[compareImage_path]['crop']:
                                if c not in to_add_json_compare[compareImage_path]:
                                    to_add_json_compare[compareImage_path].append(c)
    fig = plt.figure(figsize=(12, 8))
    im_crop = load_image_into_numpy_array(cropped_image[1])
    sub_plot = fig.add_subplot(len(all_images) / 4 + 2, 4, 1)
    plt.imshow(im_crop)
    sub_plot.axis('off')
    sub_plot.set_title('Current object box')
    idx = 0
    for key, value in all_images.items():
        im = Image.open(key)
        im_np = load_image_into_numpy_array(im)
        sub_plot = fig.add_subplot(len(all_images) / 4 + 2, 4, idx + 4 + 1)
        sub_plot.set_title('Similarity: {0:.3f}'.format(value))
        sub_plot.axis('off')
        if key in correct_images:
            plt.imshow(im_np)
        else:
            plt.imshow(im_np, alpha=0.3)
        idx += 1

    plt.show()


COMPARE_JSON_PATH = 'compare.json'
with open(COMPARE_JSON_PATH, 'r') as fin:
    json_compare = json.load(fin)

json_compare['results'] = []
for image_path, classes in to_add_json_compare.items():
    json_compare['results'].append({"imagePath": image_path, "classes": classes})
    print(image_path + ' --> ' + ','.join(classes))

with open(COMPARE_JSON_PATH, 'w') as fout:
    json.dump(json_compare, fout)



end = time.time()
print('Elapsed time creating db:' + str(enddb - startdb))
print('Elapsed time finding similar imgs:' + str(end - find))
print('Elapsed time:' + str(end - start))



