
Download [ST-VQA dataset](https://rrc.cvc.uab.es/?ch=11) and place all files into this folder. The folder tree should be like this:


```
data/ST-VQA/coco-text
data/ST-VQA/icdar
data/ST-VQA/IIIT_text
data/ST-VQA/imageNet
data/ST-VQA/VisualGenome
data/ST-VQA/vizwiz
```

You may want to replace the ```coco-text``` folder (which has images downscaled to 256x256) with the one provided in this repository (original COCO-Text images):


```
rm -Rf data/ST-VQA/coco-text
cp -Rf ../STVQA_v2/coco-text_images data/ST-VQA/coco-text
```
