# Facial-Expression-Recognition-Emonet

Official implementation of the paper _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 [[1]](#Citation).
Work done in collaboration between Samsung AI Center Cambridge and Imperial College London.

Please find a full-text, view only, version of the paper [here](https://rdcu.be/cdnWi).

The full article is available on the [Nature Machine Intelligence website](https://www.nature.com/articles/s42256-020-00280-0).

[Demo] Discrete Emotion + Continuous Valence and Arousal levels      |  [Demo] Displaying Facial Landmarks
:-------------------------------------------------------------------:|:--------------------------------------:
<img src='images/emotion_only.gif' title='Emotion' style='max-width:600px'></img>  |  <img src='images/emotion_with_landmarks.gif' title='Emotion with landmarks' style='max-width:600px'></img>


## Our Project
Introduces a novel deep neural network for analyzing facial affect in naturalistic conditions. 
Integrates face alignment and jointly estimates categorical and continuous emotions in a single pass.

In our project we used the emonet while using a transfer learning technique to train the model on our dataset. The dataset was different from the data that the original model was trained on. The dataset was a collection of images of people with different facial expressions - FER2013 [Link](https://paperswithcode.com/dataset/fer2013).

The dataset was labeled for 7 emotions which is different from the pretrained models, so we had to modify the model to output 7 classes instead of 8 or 5, and focus on the categorical emotions only.

Therefore, we created a train.py file that trains the model on the smaller dataset which had to be augmented.. 

### WIP (need to add more details about the project)

and saves the model to be used later for testing.

#### Class number to expression name

The mapping from class number to expression is as follows.

```
For 8 emotions :

0 - Neutral
1 - Happy
2 - Sad
3 - Surprise
4 - Fear
5 - Disgust
6 - Anger
7 - Contempt
```

```
For 5 emotions :

0 - Neutral
1 - Happy
2 - Sad
3 - Surprise
4 - Fear
```

## Citation

If you use this code, please cite:

```
@article{toisoul2021estimation,
  author  = {Antoine Toisoul and Jean Kossaifi and Adrian Bulat and Georgios Tzimiropoulos and Maja Pantic},
  title   = {Estimation of continuous valence and arousal levels from faces in naturalistic conditions},
  journal = {Nature Machine Intelligence},
  year    = {2021},
  url     = {https://www.nature.com/articles/s42256-020-00280-0}
}
```

[1] _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 

## License

Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0 International Licence (CC BY-NC-ND) license.
