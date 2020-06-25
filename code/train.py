import time
import numpy as np
import tensorflow as tf
from models.attention import *
from utils.data_loader import *
from utils.utils import *

gt_file_train = 'data/stvqa_train.json'
image_path    = 'data/ST-VQA/'
yolo_file     = 'models/bin/yolov3_coco.pb'
fasttext_file = 'models/bin/wiki-news-300d-1M-subword.bin'

rnn_size  = 256                        # LSTM number of hidden nodes in each layer
rnn_layer = 2                          # number of the LSTM layer
img_size  = 608                        # yolo input img size (608, 608)
img_feature_shape = (38,38,512)        # yolo output
txt_feature_shape = (38,38,300)        # fasttext grid embedding
dim_image = img_feature_shape[2]       # size of visual features
dim_hidden = 1024                      # size of the common embedding vector
dim_attention = 512                    # size of attention embedding
maxlen = 25                            # question max len
text_embedding_dim = 300               # size of textual features, fasttext dimension

# Train Parameters setting
lr = 0.0003
decay_factor = 0.99997592083
batch_size = 32
n_epochs = 100

# Check point
checkpoint_path = 'models/ckpt/'

sess = tf.Session()


print_info('Loading YOLO model...')
yolo_inout = ["input/input_data:0", "conv61/Conv2D:0"]
input_images, image_features  = yolo_read_pb_return_tensors(yolo_file, yolo_inout)
print_ok('Done!\n')


print_info('Building Attention model ...')
attention_model = Attention(
                  rnn_size = rnn_size,
                  rnn_layer = rnn_layer,
                  batch_size = batch_size,
                  dim_image = (38,38,img_feature_shape[2]+txt_feature_shape[2]),
                  dim_hidden = dim_hidden,
                  dim_attention = dim_attention,
                  max_words_q = maxlen,  
                  text_embedding_dim = text_embedding_dim,
                  drop_out_rate = 0.5,
		  training=True)

txt_features = tf.placeholder(tf.float32, [batch_size, txt_feature_shape[0], txt_feature_shape[1], txt_feature_shape[2]])
question = tf.placeholder(tf.float32, [batch_size, maxlen, text_embedding_dim])

attention = attention_model.build_model(image_features, txt_features, question)
print_ok('Done!\n')

attention_trainable_var_list = tf.trainable_variables()

labels = tf.placeholder(tf.float32, shape=(batch_size,38,38)) # TODO do not hardcode!

loss = tf.losses.sigmoid_cross_entropy(labels, attention)

# Optimization
learning_rate = tf.placeholder(tf.float32)
with tf.variable_scope('Optimizer'):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=attention_trainable_var_list)

sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=10, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

# load fasttext model
print_info('Loading FastText model...')
fasttext_model = fasttext.load_model(fasttext_file)
print_ok('Done!\n')

data_generator = STVQADataGenerator(gt_file_train, maxlen, image_path, img_size, fasttext_model, batch_size=batch_size)

for epoch in range(1,n_epochs+1):
  for it in range(data_generator.len()):

    s_time = time.time()
    batch = data_generator.next()
    #for t in batch: print t.shape

    _, l = sess.run([train_step,loss], feed_dict={input_images: batch[0], txt_features: batch[1], labels: batch[3], question: batch[2], learning_rate: lr})
    print_info('Epoch: %d/%d - iter: %d/%d - loss: %f - time: %f s.\n'%(epoch, n_epochs, it, data_generator.len(), l, time.time()-s_time))

    # decrease lr every 50 epochs
    lr = lr*decay_factor
    print_info("Decreassed learning rate to lr *= decay_factor = %f\n"%lr)

  if (epoch%5 == 0): 
        # Save the variables to disk on every epoch.
        save_path = saver.save(sess, checkpoint_path+"saved_model_conv_1x1_epoch"+str(epoch)+".ckpt")
        print_ok("Model saved in file: %s\n" % save_path)

