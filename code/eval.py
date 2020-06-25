import time, sys
import numpy as np
import tensorflow as tf
from models.attention import *
from utils.data_loader import *
from utils.utils import *

gt_file_test = 'data/stvqa_eval.json'
image_path = 'data/ST-VQA/'
yolo_file  = 'models/bin/yolov3_coco.pb'
fasttext_file = 'models/bin/wiki-news-300d-1M-subword.bin'

rnn_size  = 256                        # LSTM number of hidden nodes in each layer
rnn_layer = 2                          # number of the LSTM layer
img_size  = 608                        # yolo input img size (608, 608)
img_feature_shape = (38,38,512)        # yolo output
txt_feature_shape = (38,38,300)        # fasttext grid embedding
dim_image = img_feature_shape[2]       # size of visual features
dim_attention = 512                    # size of attention embedding
batch_size = 55                        # batch_size for each iteration
dim_hidden = 1024                   # size of the common embedding vector
maxlen = 25                         # question max len
text_embedding_dim = 300            # size of textual features, fasttext dimension

# Check point
if len(sys.argv) < 2:
    print_err('ERR: I need the ckeckpoint!\n')
    quit()
checkpoint_path = sys.argv[1]

sess = tf.Session()

return_elements = ["input/input_data:0", "conv61/Conv2D:0"]
print_info('Loading YOLO model...')
input_images, image_features  = yolo_read_pb_return_tensors(yolo_file, return_elements)
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
                  drop_out_rate = 0.,
                  training=False)

txt_features = tf.placeholder(tf.float32, [batch_size, txt_feature_shape[0], txt_feature_shape[1], txt_feature_shape[2]])
question = tf.placeholder(tf.float32, [batch_size, maxlen, text_embedding_dim])

output = attention_model.build_model(image_features, txt_features, question)
print_ok('Done!\n')

# Add ops to save and restore all the variables.
print_info('Loading weights from '+checkpoint_path+'...')
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
saver.restore(sess,checkpoint_path)
print_ok('Done!\n')

# load fasttext model
print_info('Loading FastText model...')
fasttext_model = fasttext.load_model(fasttext_file)
print_ok('Done!\n')

data_generator = STVQADataGenerator(gt_file_test, maxlen, image_path, img_size, fasttext_model, batch_size=batch_size, training= False, shuffle=False)

count = 0
eval_out = []

for it in range(data_generator.len()):

    batch = data_generator.next()
    this_batch_size = len(batch[0])
    count += this_batch_size

    if this_batch_size != batch_size: # deal with last batch
        batch[0] = np.resize(np.array(batch[0]), (batch_size, img_size, img_size, 3))
        batch[1] = np.resize(batch[1], (batch_size, 38, 38, text_embedding_dim))
        batch[2] = np.resize(batch[2], (batch_size, maxlen, text_embedding_dim)) 

    out = sess.run(output, feed_dict={input_images: batch[0], txt_features: batch[1], question: batch[2]})

    batch_ocr    = batch[4]
    gt_ids       = batch[7]

    for b in range(this_batch_size):

        one_pred  = []
        all_pred  = []
        cmb_pred  = []
        best_prob = 0.
        for i in range(38):
          for j in range(38):
            if out[b,i,j] > 0.5 and batch_ocr[b,i,j] not in all_pred and batch_ocr[b,i,j] != '':
              all_pred.append(batch_ocr[b,i,j])
            if out[b,i,j] > best_prob:
              best_prob = out[b,i,j]
              one_pred = [batch_ocr[b,i,j]]
            if out[b,i,j] > 0.95 and batch_ocr[b,i,j] not in cmb_pred and batch_ocr[b,i,j] != '':
              cmb_pred.append(batch_ocr[b,i,j])
            
        prediction = ''
        if len(cmb_pred) > 0:
          prediction = ' '.join(cmb_pred)
        else:
          prediction = ' '.join(one_pred)
    
        printProgressBar(int(count), int(data_generator.len()*batch_size))
        eval_out.append({'answer': prediction, 'question_id': int(gt_ids[b])})


with open('eval_out.json', 'w') as f:
    json.dump(eval_out, f)
