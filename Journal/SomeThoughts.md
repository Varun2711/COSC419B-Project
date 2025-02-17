# Mark

  

## Feb 17, 2025

  

### Remove Legibility Classifier:

  

- Most time-consuming part
- Didn't remove too many images. After filtering, there were still about 100,000 images left.
- As a supplement, when the model will predict the value of each category [0.2, 0.4, 0.1, …], the category with the maximum value must be greater than a threshold (e.g. 0.6). (Alex's idea)

  

### Text Recognition:

  

- In the paper they use a Transformer-based autoregressive sequence model _(Scene Text Recognition with Permuted Autoregressive Sequence Models)_.
- Could be replaced using a simpler/faster model
-- _JEDE: Universal Jersey Number Detector for Sports (Also cited by this paper)_
-- _Multi-task Learning for Jersey Number Recognition in Ice Hockey (Also cited by this paper)_
-- _Automatic Team Assignment and Jersey Number Recognition in Football Videos_
-- _Generalized Jersey Number Recognition Using Multi-task Learning With Orientation-guided Weight Refinement_
-- Handwriting recognition model (Alex's idea)
-- A3 model?