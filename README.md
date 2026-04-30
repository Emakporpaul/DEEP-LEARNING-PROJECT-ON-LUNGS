### 5. Run the Streamlit app
```bash
streamlit run src/app.py
```

### 6. Or run the full training notebook
```bash
jupyter notebook notebooks/deep_learning_lung_project.ipynb
```

---

## 📦 Dataset

**COVID-19 Radiography Database**  
Tawsifur Rahman, Amith Khandakar, et al.  
Available on [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

- 13,808 total images
- 3,616 COVID-19 positive
- 10,192 Normal
- Image format: PNG, 299×299px

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| TensorFlow 2.21 | Model training and inference |
| Keras | High-level neural network API |
| MobileNetV2 | Pretrained base model (ImageNet) |
| OpenCV | Image processing and Grad-CAM overlay |
| Scikit-learn | Evaluation metrics |
| Streamlit | Web application deployment |
| Matplotlib / Seaborn | Visualizations |
| Pandas / NumPy | Data manipulation |

---

## 📋 Notebook Pipeline

The training notebook covers the full ML pipeline:

1. Exploratory Data Analysis — class distribution, sample images
2. Data loading with 70/15/15 train/val/test split
3. Preprocessing — normalization, augmentation pipeline
4. Custom CNN baseline — architecture, training, evaluation
5. MobileNetV2 Phase 1 — frozen base, head training
6. MobileNetV2 Phase 2 — selective fine-tuning
7. Training curves — accuracy, loss, AUC across both phases
8. Full evaluation — confusion matrix, ROC curve, F1, AUC
9. Grad-CAM explainability — spatial attention visualization
10. Model comparison and saving

---

## 🔮 Future Work

- Extend to multi-class classification — COVID, Pneumonia, Tuberculosis, Normal
- Integrate DICOM support for CT scan volumetric analysis
- Explore Vision Transformers (ViT) for improved performance
- Add confidence calibration for clinical deployment safety
- Build REST API with FastAPI for integration with hospital systems

---

## 👨‍💻 Author

**Emakpor Paul**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-paulemakpor-blue)](https://www.linkedin.com/in/paulemakpor/)
[![GitHub](https://img.shields.io/badge/GitHub-Emakporpaul-black)](https://github.com/Emakporpaul)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> **Medical Disclaimer:** This tool is for educational and research purposes 
> only. It is not a substitute for professional medical diagnosis. Always 
> consult a qualified medical professional for clinical decisions.