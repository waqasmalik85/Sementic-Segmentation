import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import scipy
import numpy as np
from functools import partial
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    default_graph = tf.get_default_graph()
    vgg_input_tensor = default_graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
      conv_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    output = tf.layers.conv2d_transpose(conv_layer7, num_classes, 4, 2, 'same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    output = tf.add(output, conv_layer4)  
    conv_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, 'same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)) 
    output = tf.add(output, conv_layer3)  

    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, 'same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))  

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, optimizer, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
      for epoch in range(epochs):
        for i, (img, label) in enumerate(get_batches_fn(batch_size)):
            f_dict = {input_image: img, correct_label: label, keep_prob: 0.5, learning_rate: 1e-4}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=f_dict)
            print('Loss for batch {0} of iteration {1}: {2}'.format(i, epoch, loss))
tests.test_train_nn(train_nn)


def process_image(image, sess, logits, keep_prob, input_image, image_shape):
    
    image = scipy.misc.imresize(image, image_shape)
    
    im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
        

    label_index = np.argmax(im_softmax, axis=2)

    value_fill_1 = label_index.copy()
    value_fill_1.fill(1)
    value_fill_2 = label_index.copy()
    value_fill_2.fill(2)
    value_fill_3 = label_index.copy()
    value_fill_3.fill(3)

    segmentation1 = np.equal(label_index, value_fill_1)
    segmentation2 = np.equal(label_index, value_fill_2)
    segmentation3 = np.equal(label_index, value_fill_3)
    
    segmentation1 = segmentation1.reshape(image_shape[0], image_shape[1], 1)
    segmentation2 = segmentation2.reshape(image_shape[0], image_shape[1], 1)
    segmentation3 = segmentation3.reshape(image_shape[0], image_shape[1], 1)
    mask1 = np.dot(segmentation1, np.array([[0, 255, 0, 127]]))
    mask1 = scipy.misc.toimage(mask1, mode="RGBA")
    mask2 = np.dot(segmentation2, np.array([[0, 0, 255, 127]]))
    mask2 = scipy.misc.toimage(mask2, mode="RGBA")
    mask3 = np.dot(segmentation3, np.array([[255, 242, 0, 127]]))
    mask3 = scipy.misc.toimage(mask3, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask1, box=None, mask=mask1)
    street_im.paste(mask2, box=None, mask=mask2)
    street_im.paste(mask3, box=None, mask=mask3)
    
    return np.array(street_im)

def run():
    re_train = False
    predict_image = True
    process_video = False
    
    tf.reset_default_graph()
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        
        # hyper parameters
        
        epochs = 30
        batch_size = 8
        
        # place holders
        
        correct_label = tf.placeholder(tf.int32, shape=[None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        # TODO: Build NN using load_vgg, layers, and optimize function
        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        
        sess.run(tf.global_variables_initializer())

        if re_train:
        
            saver = tf.train.Saver()
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input, correct_label,
                    vgg_keep_prob, learning_rate)

            saver.save(sess, 'segmentation_model')
            
        if predict_image:
        
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint('./'))
            scipy.misc.imshow(process_image(scipy.misc.imread("./data/data_road/training/image_2/uu_000040.png"),
                                           sess, logits, vgg_keep_prob, vgg_input, image_shape))
        
        if(process_video):
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint('./'))
            Output_video = 'project_video_seg.mp4'
            Input_video = 'project_video.mp4'
            
            partial_process_image = partial(process_image,  sess=sess, logits=logits, keep_prob=vgg_keep_prob,
                                            input_image=vgg_input, image_shape=image_shape)

            video_output = Output_video
            clip1 = VideoFileClip(Input_video)
            
            video_clip = clip1.fl_image(partial_process_image) #NOTE: this function expects color images!!
            video_clip.write_videofile(video_output, audio=False)
        

if __name__ == '__main__':
    run()