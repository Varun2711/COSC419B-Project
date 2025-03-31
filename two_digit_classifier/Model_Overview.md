### **Model Overview: Two-Digit Jersey Number Classification**
This task involves recognizing jersey numbers from images, where numbers can be:
- **Single-digit** (e.g., `7`, `9`)  
- **Double-digit** (e.g., `10`, `23`)  

The model is designed to handle both cases by treating them as **two separate classification tasks**:
1. **First digit**: Predicts `0-9` (always present).
2. **Second digit**: Predicts `0-9` **or** an `"empty"` class (indicating a single-digit number).  

---

### **Key Components of the Model**
#### **1. Backbone: Pretrained ResNet**
- The model uses a **pretrained ResNet** (default: `ResNet18`) as its backbone.  
  - Pretrained on **ImageNet**, so it already understands basic visual features (edges, textures, shapes).  
  - The last fully connected (FC) layer is **removed**, keeping only the feature extractor.  

#### **2. Custom Classification Heads**
Two new FC layers are added:
- **First-digit head**: Predicts `0-9` (10 classes).  
- **Second-digit head**: Predicts `0-9` **+** `"empty"` (11 classes).  

#### **3. How Predictions Work**
- The backbone extracts features from the input image.  
- These features are fed into both heads:  
  - If the **second digit** predicts `"empty"`, the number is single-digit (e.g., `7`).  
  - Otherwise, the outputs are combined (e.g., `1` + `0` → `10`).  

---

### **How the Pretrained Model is Adapted**
1. **Transfer Learning**  
   - The pretrained ResNet (trained on ImageNet) is **frozen** initially (optional) or fine-tuned.  
   - Only the **new classification heads** are trained from scratch.  

2. **Training Process**  
   - **Loss Function**: Cross-entropy loss for both digits (`loss1 + loss2`).  
   - **Optimizer**: Adam (common for deep learning).  
   - **Input**: Images resized to `224x224` and normalized (standard for ResNet).  

3. **Handling Confidence**  
   - During testing, low-confidence predictions (`< 0.6`) are discarded to reduce errors.  
   - For tracklets (multiple images of the same number), predictions are **aggregated** by confidence scores.  

---

### **Why This Design?**
- **Efficiency**: Pretrained ResNet provides strong feature extraction without training from scratch.  
- **Flexibility**: Handles both single and double-digit numbers seamlessly.  
- **Robustness**: Confidence thresholds and tracklet aggregation improve real-world reliability.  

This approach balances accuracy and practicality for jersey number recognition in sports.
