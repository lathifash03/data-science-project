# Data Science Projects

Coursework projects from my Informatics Engineering degree at Institut Teknologi Sepuluh Nopember (ITS). Each folder has its own notebook and README with the data, method, results, and known limitations.

These are course projects. My research work (thesis pipeline and agentic knowledge-graph verification) is in separate repositories, linked at the bottom.

| Project | Task | Methods | Main result |
|---|---|---|---|
| [fraud-detection](fraud-detection/) | Fraud detection on a highly imbalanced transaction dataset | Feature engineering, SMOTE, Logistic Regression, Decision Tree | Decision Tree: PR-AUC 0.86, F1 0.77 on 22 fraud cases in the test set |
| [stock-transformer](stock-transformer/) | Stock price forecasting and up/down classification | Transformer, hyperparameter tuning, SMOTE, attention variant | Did not beat a trivial baseline (documented) |
| [happiness-clustering](happiness-clustering/) | Resident satisfaction survey: classification and segmentation | KNN, K-Means | KNN accuracy 0.70 (67 test samples); weak cluster structure |
| [mlbb-sentiment](mlbb-sentiment/) | Sentiment analysis of Indonesian Google Play reviews | VADER labels, TF-IDF, SMOTE, Random Forest, SVM, Naive Bayes | Labels are weak (see limitations) |

## Running the notebooks

Each notebook was written in Google Colab. Open it in Colab, or run it locally:

```bash
pip install -r requirements.txt
jupyter notebook
```

Datasets are not stored in this repository. Each project README says where the data comes from.

## Related repositories

- Thesis demo: https://github.com/lathifash03/ta_demo
- Agentic verification over a knowledge graph: https://github.com/lathifash03/kg-agentic
- Portfolio: https://lathifash03.github.io

## Author

Lathifah Sahda. LinkedIn: https://linkedin.com/in/lathifah-sahda
