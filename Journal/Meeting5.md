# Group meeting 5

## February 10, 2025

### Progress:

- Explored alternative models like JEDE, ResNet-34, and YOLOv5 for jersey number recognition
- Be able to run python3 setup.py at the local environment
- Downloaded the trained model and replaced the one present in the `reid\centroids-reid\models` path
- Discussed using a lightweight model pre-trained on the MNIST dataset
  - Output layer mismatch: Identify the issue of MNIST's 10-class output vs our 100-class requirement
  - Input distribution shift: Discussed how to preprocessed images to match the MNIST format.

### Next week goals:

- Evaluate the limitations of lightweight models and consider using deeper models like ResNet or VGG
- Explore decomposing numbers into digits for multi-task learning.
- Figure out the drawbacks of the current jersey-number-pipeline and find areas of improvement
