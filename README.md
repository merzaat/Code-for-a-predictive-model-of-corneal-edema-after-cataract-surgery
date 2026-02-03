No problem. Here is the **GitHub README.md** content for your project in Markdown format, which you can copy directly into your repository.

# Ophthalmic Risk-Prediction Software for Post-Operative Corneal Edema

### An Interpretable ML Framework for Clinical Decision Support (CDSS)
<img width="1920" height="946" alt="image" src="https://github.com/user-attachments/assets/eba5b072-3944-4ca9-82e0-3b943cd989c2" />


## 👁️ Overview

This repository contains the core logic and GUI architecture for a **Machine Learning-driven risk prediction system** designed to forecast clinically significant **corneal edema (Grade ≥ 3)** following phacoemulsification cataract surgery.

Unlike traditional "black-box" models, this framework leverages **Explainable AI (XAI)** to provide surgeons with a clear biomedical rationale for each risk assessment, enabling personalized surgical planning and energy setting optimization.

## 🚀 Key Features

* **Real-time Risk Stratification**: Predicts high-risk patients (associated with  endothelial cell loss) during preoperative planning.
* **Explainable Rationale**: Integrated **SHAP** visualizations to show the synergistic impact of surgical energy loads versus anatomical protective factors (e.g., **ACD**, Chamber Volume).
* **User-Friendly GUI**: A specialized dashboard designed for clinician use, bypassing the need for coding knowledge in the operating theater.
* **Robust Feature Selection**: Utilizes a dual **Boruta + LASSO** strategy to handle high-dimensional noise in real-world clinical datasets.

## 🔬 Methodology & Performance

Based on a retrospective cohort of **307 eyes** and 36 perioperative variables, the model was refined through:

* **Algorithm**: Gradient Boosting (GB) optimized via **Optuna** (Bayesian Optimization).
* **Performance**: Achieved a state-of-the-art **AUC of 0.835**, significantly outperforming traditional linear risk-assessment benchmarks.
* **Core Predictors**: Identified 7 high-impact factors including Age, **Anterior Chamber Depth (ACD)**, and Effective Phaco Time.

## 🛠️ Tech Stack

* **Languages**: Python (NumPy, Pandas, Scikit-learn, XGBoost)
* **Interpretability**: SHAP (SHapley Additive exPlanations)
* **Hyperparameter Tuning**: Optuna
* **Interface**: Tkinter / PyQt (GUI Framework)

## 🏥 Clinical Significance

The software is currently undergoing **internal clinical validation**. By identifying high-risk individuals before surgery, clinicians can:

1. **Individualize** ultrasonic energy and fluidics settings.
2. **Optimize** perioperative medication protocols.
3. **Enhance** patient counseling for long-term corneal health expectations.

## 📬 Contact & Collaboration

Developed by **Mierzhati Miershali**

*Clinician and Researcher in Ophthalmic AI*

📧 [merzat.mersali@foxmail.com](mailto:merzat.mersali@foxmail.com)

🎓 MSc in Ophthalmology, Xinjiang Medical University

