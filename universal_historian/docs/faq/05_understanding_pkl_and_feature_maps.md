# FAQ 05: What are .pkl files and Feature Maps?

### ❓ The Question:
What is a `.pkl` file, and why do I have two of them for the universal model?

### 🔬 The Technical Answer:
**`.pkl`** stands for **Pickle.** It is the way Python "Freezes" a live object so it can be saved to your hard drive.

1.  **The Brain (`universal_historian_v1.pkl`):** 
    *   This is the **Trained Model.** It is a complex binary object containing the 200 decision trees and the weights learned from the data.
2.  **The Reference Map (`universal_features_map.pkl`):** 
    *   This is not a model. It is just a **Python List** of the 13 feature names.
    *   **Why you need it:** AI models are "Position Sensitive." If you feed in the `annual_inc` where it expects the `dti`, it will give you a wrong answer. The Map ensures your new applicant's data is **Sorted** into the correct order before it hits the Brain.

**Analogy:** One is the **Key** (The Model) and the other is the **Key Pattern** (The Map) showing how the teeth should be cut.
