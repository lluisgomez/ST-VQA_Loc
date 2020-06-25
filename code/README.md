### requirements

- python2.7
- tensorflow==1.10.1
- fasttext==0.9.1
- opencv-python==3.4.3.18

### setup

1) Clone the repository:

```
git clone https://github.com/lluisgomez/ST-VQA_Loc
```

2) Download YOLOv3 and FastText models:

```
cd ST-VQA_Loc/code/models/bin
wget https://stvqa-loc-bin.s3.us-east-2.amazonaws.com/yolov3_coco.pb
wget https://stvqa-loc-bin.s3.us-east-2.amazonaws.com/wiki-news-300d-1M-subword.bin
```

3) Download [ST-VQA dataset](https://rrc.cvc.uab.es/?ch=11) and place all files into the ```ST-VQA_Loc/data/ST-VQA/``` folder.

### inference and evaluation of pre-trained model

Run the following command to use the provided checkpoint for inference on the ST-VQA test set. It will create a file ```eval_out.json``` with question_ids and answers as required by the [ST-VQA evaluation server](https://rrc.cvc.uab.es/?ch=11).

```
cd ST-VQA_Loc/code/
python test.py models/ckpt/saved_model_conv_1x1_epoch75.ckpt
```

### train

Run the following command to train the model on the ST-VQA train set:

```
cd ST-VQA_Loc/code/
python train.py
```

If you want to train the model on another dataset you must create json data files with same structure as ```data/stvqa_train.json``` and ```data/stvqa_eval.json```. They are basically lists of json objects with the following structure:

```
{u'ans_bboxes': [{u'bbox': [313, 322, 331, 353], u'text': u'susan'}],
 u'answer': [u'susan'],
 u'file_path': u'coco-text/COCO_train2014_000000347021.jpg',
 u'ocr_bboxes': [{u'bbox': [167, 240, 196, 338], u'text': u'conversations'},
  {u'bbox': [182, 222, 200, 238], u'text': u'on'},
  {u'bbox': [189, 236, 217, 332], u'text': u'consciousness'},
  {u'bbox': [274, 294, 281, 304], u'text': u'wat'},
  {u'bbox': [266, 328, 274, 341], u'text': u'in'},
  {u'bbox': [264, 344, 270, 350], u'text': u'th'},
  {u'bbox': [313, 322, 331, 353], u'text': u'susan'},
  {u'bbox': [321, 276, 342, 320], u'text': u'blackmore'}],
 u'question': [u'what', u'is', u'the', u'book', u'authors', u'first', u'name'],
 u'question_id': 125}
```

In the case of training data the field ```'question_id'``` is not necessary for training, while in the case of test data the fields ```'answer'``` and ```'ans_bboxes'``` are not necessary to create the ```eval_out.json``` file.

