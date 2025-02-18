# Mark

  

## Feb 17, 2025

  

### Remove Legibility Classifier:

  

- Most time-consuming part
- Didn't remove too many images. After filtering, there were still about 100,000 images left.
- As a supplement, when the model predicts each category's value [0.2, 0.4, 0.1, …], the category with the maximum value must be greater than a threshold (e.g., 0.6). (Alex's idea)

  

### Text Recognition:

  

* The paper uses a Transformer-based autoregressive sequence model _(Scene Text Recognition with Permuted Autoregressive Sequence Models)_.
* Could be replaced using a simpler/faster model:
  
  * _JEDE: Universal Jersey Number Detector for Sports (Also cited by this paper):_ [https://ieeexplore.ieee.org/abstract/document/9810931](https://ieeexplore.ieee.org/abstract/document/9810931)
    * First predicts digit centers and bounding box sizes using pose-guided heatmap regression.
    * Then an MLP classifier determines the number of digits (0, 1, or 2).
    * Finally, the digit branch classifies each detected digit (0-9), and a post-processing step pairs digits together if necessary to generate the final jersey number.
  
  * _Multi-task Learning for Jersey Number Recognition in Ice Hockey (Also cited by this paper):_ [https://ieeexplore.ieee.org/abstract/document/9810931](https://dl.acm.org/doi/abs/10.1145/3475722.3482794)
    * built on ResNet-34, which produces 512-dimensional feature representations.
    * These 512-dimensional features are then fed into three separate fully connected layers, each producing different probability outputs: Holistic Representation Layer (81-dimensional), First-Digit Layer (11-dimensional), Second-Digit Layer (another 11-dimensional)
  
  * _Automatic Team Assignment and Jersey Number Recognition in Football Videos_
  
  * _Generalized Jersey Number Recognition Using Multi-task Learning With Orientation-guided Weight Refinement:_ [https://arxiv.org/abs/2406.01033](https://arxiv.org/abs/2406.01033)

    * Using ResNet50 as the backbone, which extracts 512-dimensional features.
    * After feature extraction, the model has four parallel fully connected layers, each corresponding to a different task: overall probability, probability of tens-digit numbers, probability of ones-digit numbers, and probability of the number of digits.
    * Uses human body orientation computed using HRNet to improve digit recognition.
  
  * Handwriting recognition model (Alex's idea)
  
  * A3 model?
