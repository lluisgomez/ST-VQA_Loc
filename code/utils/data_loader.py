import os
import json
import numpy as np
import cv2
import fasttext

from utils import *
from yolo_utils import *

import tensorflow as tf

class STVQADataGenerator:

    def __init__(self, gt_file, maxlen, image_path, input_size, fasttext_model, batch_size=32, training=True, shuffle=True):

        self.training   = training
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.maxlen     = maxlen
        self.image_path = image_path
        self.fasttext_model = fasttext_model
        self.fasttext_dim = self.fasttext_model.get_dimension()
        self.curr_idx   = 0
        self.input_size = input_size

        # load gt file
        print_info('Loading GT file...')
        with open(gt_file) as f:
          self.gt = json.load(f)
        print_ok('Done!\n')
        # TODO filter questions by maxlen?
    

    def len(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.gt) / self.batch_size))

    def next(self):
        # select next batch idxs
        if self.shuffle: 
          batch_idxs  = np.random.choice(len(self.gt), self.batch_size)
        else:
          if self.curr_idx + self.batch_size > len(self.gt): self.curr_idx = 0
          batch_idxs    = range(self.curr_idx, self.curr_idx+self.batch_size)
          self.curr_idx = (self.curr_idx+self.batch_size) % len(self.gt)

        batch_x_image = []
        batch_x_textual   = np.zeros((self.batch_size, 38, 38, self.fasttext_dim)) # TODO do not hardcode contants
        batch_x_questions = np.zeros((self.batch_size, self.maxlen, self.fasttext_dim))
        batch_y = np.zeros((self.batch_size, 38, 38), dtype=np.int8)
        if not self.training:
          batch_ocr = np.chararray((self.batch_size, 38, 38), itemsize=35)
          batch_ocr[:] = ''
          gt_questions = []
          gt_answers   = []
          gt_ids       = []

        # foreach question in batch
        for i,idx in enumerate(batch_idxs):

            # load image
            #print os.path.join(self.image_path, self.gt[idx]['file_path'])
            image = cv2.imread(os.path.join(self.image_path, self.gt[idx]['file_path']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

            # get fasttext vectors and bboxes for all ocr_tokens
            gt_boxes = [w['bbox'] for w in self.gt[idx]['ocr_bboxes']]
            gt_text_vectors = [self.fasttext_model.get_word_vector(w['text']) for w in self.gt[idx]['ocr_bboxes']]
            gt_texts = [w['text'] for w in self.gt[idx]['ocr_bboxes']]
            # store indexes of those bboxes wich are the answer
            gt_ans_boxes = [w['bbox'] for w in self.gt[idx]['ans_bboxes']]
            gt_ans_idxs  = [gt_boxes.index(b) for b in gt_ans_boxes]
            gt_boxes = np.array(gt_boxes)

            # TODO data augmentation?

            # preprocess image 
            if len(gt_boxes) > 0:
              image_data,gt_boxes = yolo_image_preporcess(image, [self.input_size, self.input_size], gt_boxes=gt_boxes)
            else:
              image_data = yolo_image_preporcess(image, [self.input_size, self.input_size])
              gt_boxes = np.array(())
            batch_x_image.append(image_data)

            # assign fasttext vectors to cells in a 38x38 grid
            for w in range(gt_boxes.shape[0]):
              cell_coords = gt_boxes[w,:] / 16 # TODO do not hardcode contants
              batch_x_textual[i, cell_coords[1]:cell_coords[3]+1, cell_coords[0]:cell_coords[2]+1, :] = gt_text_vectors[w]
              if w in gt_ans_idxs:
                  batch_y[i, cell_coords[1]:cell_coords[3]+1, cell_coords[0]:cell_coords[2]+1] = 1
              if not self.training:
                  batch_ocr[i, cell_coords[1]:cell_coords[3]+1, cell_coords[0]:cell_coords[2]+1] = gt_texts[w]

            # question encode with fasttext
            question = self.gt[idx]['question']
            for w in range(self.maxlen-len(question), self.maxlen):
              batch_x_questions[i, w, :] = self.fasttext_model.get_word_vector(question[w-(self.maxlen-len(question))])

            # if not training return gt for evaluation
            if not self.training:
              gt_questions.append(' '.join(self.gt[idx]['question']))
              gt_answers.append(' '.join(self.gt[idx]['answer']))
              if 'question_id' in self.gt[idx].keys():
                gt_ids.append(self.gt[idx]['question_id'])

        if self.training:
          return [batch_x_image, batch_x_textual, batch_x_questions, batch_y]
        else:
          return [batch_x_image, batch_x_textual, batch_x_questions, batch_y, batch_ocr, gt_questions, gt_answers, gt_ids]


