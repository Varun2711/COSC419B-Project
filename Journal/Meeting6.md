# Group meeting 6

## February 17, 2025

### Progress:

- Finally ran the `jersey-number-pipeline` repository on the machines of some group members.
- Noticed that the inference time is very long: 8 Hours and 36 minutes for one of the runs.
- Discussed and decided to remove the legibility classifier, as it was time consuming and not very impactful
- Decided to use ResNet as the backbone for our model, with parallel fully connected layers for multi-task learning.

### Next week goals:

- Define appropriate evaluation metrics to measure the model's performance accurately.
- Begin optimizing the pipeline by removing unnecessary components and replacing the PARSeq model with ResNet
- Explore data augmentation techniques to improve model performance.
