# Word2Vec Implementation

This project implements the Skip-Gram Word2Vec model to learn dense vector representations (embeddings) of words from the BBC news dataset. These embeddings capture semantic and contextual relationships between words, enabling applications like similarity calculation, clustering, and visualization of meaningful word groupings.

## Preprocessing

- Tokenize raw text into words.
- Convert text to lowercase and clean punctuation.
- Build a vocabulary of unique words, optionally filtering rare words.

## Training

- Generate (target, context) word pairs using a sliding window.
- (Optional) Use negative sampling for efficient softmax approximation.
- Train embeddings using stochastic gradient descent, optimizing context prediction.

## Post-processing

- Save trained embeddings to `embeddings.npy`.
- Visualize embeddings in 2D with PCA to observe semantic clusters.

## Insights

This approach derives meaningful representations directly from raw text, uncovering latent semantic relationships that traditional methods fail to capture. Such embeddings are foundational for various NLP tasks including text classification, information retrieval, and recommendation systems.

## Instructions to Run the Code

### Place the Dataset  
Put `bbc_news_text_complexity_summarization.csv` in the same folder as the script file (e.g., `word2vec_from_scratch.py`).

### Install Dependencies  
Run this command if you haven't installed the libraries yet:

pip install numpy pandas matplotlib scikit-learn

### Run the Script  
Navigate to the directory and run:

python word2vec_from_scratch.py

### View Outputs  
The script will train the Word2Vec model, save embeddings as `embeddings.npy` (or `embeddings.txt`), and optionally generate visualization plots like PCA scatterplots.

### Optional  
If using Jupyter Notebook or Google Colab, run notebook cells instead of the standalone script.  
Ensure your runtime has Python 3.x installed.

## Outputs
- **Word Embeddings:** Saved as `embeddings.npy` (binary) or `embeddings.txt` (text), representing semantic vectors for each word.  
- **Visualizations (optional):** 2D plots via t-SNE or PCA saved in `results/` folder (e.g., `tsne.png`, `pca.png`).  
- **Console Output:** Training progress logs (loss per epoch) and optional similarity scores.
  ![Word2Vec Output](results/output.png)



