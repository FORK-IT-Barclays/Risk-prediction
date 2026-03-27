# FAQ 02: How does the AI "Critically" decide?

### ❓ The Question:
What is the model actually doing in the most "exact and critical" way to make a decision?

### 🔬 The Technical Answer:
The model uses **Gradient Boosting (XGBoost)** to perform **Sequential Correction.**

1.  **Iterative Improvement:** The AI builds **200 Decision Trees** one after another. Every new tree is built specifically to "fix" the mistakes made by the tree that came before it. 
2.  **Information Gain:** At every branch, the AI performs a "Greedy Search" for the exact number (like `revol_util > 91.2%`) that creates the cleanest separation between Safe and Default loans.
3.  **Scale Positive Weight:** Because Defaults are rare, we apply a **4.7x Penalty** to the model. This means if the model misses a "Defaulter," it is punished 4.7 times harder than if it misses a "Safe" person. 

**Conclusion:** The model is a hyper-sensitive "Fault Detector" that has been trained to be obsessed with finding the tiny patterns that lead to financial collapse.
