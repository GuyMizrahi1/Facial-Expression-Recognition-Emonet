# Facial-Expression-Recognition-Emonet

Official implementation of the paper _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 [[1]](#Citation).
Work done in collaboration between Samsung AI Center Cambridge and Imperial College London.

Please find a full-text, view only, version of the paper [here](https://rdcu.be/cdnWi).

The full article is available on the [Nature Machine Intelligence website](https://www.nature.com/articles/s42256-020-00280-0).

## Our Project
This report presents our work on extending the capabilities of EmoNet, 
a deep learning model for facial expression recognition, 
as described in "Estimation of continuous valence and arousal levels from faces in naturalistic conditions" (Toisoul et al., 2021). 
We explored novel enhancements by incorporating transfer learning and attention mechanisms. 
Our experiments aimed to improve the model's ability to capture salient facial features for accurate expression 
classification using attention mechanisms. 
A detailed analysis of our methods, results, and insights gained from the project is provided. 
The code and model implementations are available here in this repository on GitHub, 
and the outputs, including graphs and matrices, can be accessed on [Google Drive](https://drive.google.com/drive/folders/1frfusXOtmmxYaBml56lpZ2Lp90npFawH). 
Additionally, the model was executed in a [Google Colab notebook](https://colab.research.google.com/drive/1PU2hOrGYpgbbPH-NdP8kYq3wdbBlElgQ?authuser=0#scrollTo=drgbUr8HCGaw) 
, which is available for review.

### Enhancements and Methods
1. **Dataset and Preprocessing**: We used the FER2013 and MMAFEDB datasets, converting grayscale images to RGB and resizing all images to 256x256 pixels.
2. **Data Augmentation**: Applied random horizontal and vertical flips, and adjustments to color properties to increase dataset diversity.
3. **Model Architecture**:
   - **Baseline Model**: Transfer learning with a new classification head.
   - **EmoNet with Self-Attention**: Incorporated a dot-product self-attention mechanism.
   - **EmoNet with Multi-Head Attention**: Used a 16x16 patch size with 4 attention heads.
4. **Training and Evaluation**: Trained using the Adam optimizer, evaluated with metrics like accuracy, precision, recall, F1-score, and confusion matrices.
5. **Learning Rate Scheduler**: Implemented Cosine Annealing with Warm Restarts.
6. **Grad-CAM**: Integrated Grad-CAM for visualizing model decision-making.

### Results
- **Best Model**: The baseline model with 'Final Layer 1' achieved the highest F1 score of 52%.
- **Attention Mechanisms**: Did not improve performance due to small dataset size and data imbalance.
- **Grad-CAM**: Implementation was unsuccessful due to model complexity.

### Individual Contributions
- **Guy Mizrahi**: Implemented the train.py script, incorporated the FER2013 dataset, implemented the cosine annealing learning rate scheduler, and the dot-product self-attention mechanism.
- **Ran Weissman**: Managed the integration of the MMAFEDB dataset, implemented data augmentation techniques, and executed experiments with the EmoNet model.
- **Dolev Shaked**: Implemented the transfer learning baseline model in FerEmonet.py, multi-head attention in fer_multihead.py, and executed experiments.
- **Alon Rosenbaum**: Integrated Grad-CAM into the FerEmonet model, addressed dimension mismatches, and validated the implementation with a simpler model.


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


[Demo] Discrete Emotion + Continuous Valence and Arousal levels      |  [Demo] Displaying Facial Landmarks
:-------------------------------------------------------------------:|:--------------------------------------:
<img src='images/emotion_only.gif' title='Emotion' style='max-width:600px'></img>  |  <img src='images/emotion_with_landmarks.gif' title='Emotion with landmarks' style='max-width:600px'></img>
