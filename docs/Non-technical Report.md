# Deep Learning-Based Human Face Authenticity Detection
## Non-Technical Report

---

### **Application:** Face Forensics – AI Authenticity

---

## 1. Introduction
The rapid development of artificial intelligence has made it possible to create highly realistic human face images and other digitally generated content. In many cases, these images can appear similar to genuine photographs, making it difficult for people to determine whether an image is authentic simply by looking at it.

**Deep Learning-Based Human Face Authenticity Detection** is a project developed to address this problem by using deep learning to analyse images and provide an automated assessment of their authenticity.

The project is implemented through an application called **Face Forensics – AI Authenticity**. The application provides a simple interface through which users can upload an image and obtain an authenticity prediction. In addition to the prediction, the application provides confidence and probability information, visual explanation through Grad-CAM, forensic reporting, and analysis history.

The objective is to make the process of image authenticity analysis easier and more accessible to users without requiring them to understand the underlying deep learning technology.

---

## 2. Problem Statement
The increasing availability of AI-based image generation tools has created a challenge in identifying whether digital images are genuine or artificially generated.

A person examining an image manually may not always be able to identify subtle differences or artifacts. Therefore, an automated system can be useful for providing an initial assessment of image authenticity.

The project addresses this challenge by developing a deep learning-based system capable of analysing an image and providing an authenticity prediction.

The system does not simply display a result. It also provides supporting information such as confidence, probability values, visual explanations, and a forensic report.

---

## 3. Aim of the Project
The main aim of **Deep Learning-Based Human Face Authenticity Detection** is to develop an application that can assist in determining whether an input image is likely to be real/authentic or AI-generated.

The system is designed to provide:
- Automated authenticity detection.
- A clear prediction for the uploaded image.
- Confidence information associated with the prediction.
- AI and Real probability values.
- Visual explanation of the model's decision.
- A structured forensic report.
- A history of previous analyses.

---

## 4. Proposed Solution
The project provides a web-based graphical application called **Face Forensics – AI Authenticity**.

Instead of requiring users to interact directly with the deep learning model, the application provides a simple workflow:

$$\text{Upload Image} \longrightarrow \text{Analyse Image} \longrightarrow \text{Receive Prediction} \longrightarrow \text{Examine Explanation} \longrightarrow \text{Generate Report}$$

The application provides two analysis options:

### 4.1 Main Model
The Main Model is presented in the application as the primary model for image authenticity analysis.

### 4.2 Cross-Domain Model
The Cross-Domain Model is provided as an additional analysis option and is described in the application as being designed to generalize across different image domains and generation sources.

The two options allow the user to perform authenticity analysis through the same application interface.

### 4.3 Where to use the application

The application is available online at:

**https://group-11-ds-and-ai-lab-project.onrender.com**

Users can open this link in a web browser and begin uploading images immediately. No software installation is required. The hosted version includes the **Main Model** and **Cross-Domain Model** for everyday analysis.

Developers who need the full feature set (including additional model comparison tools) can run the application locally — instructions are in `../docs/READMEdeployment.md` and `../docs/DeveloperGuide.md`.

---

## 5. How the System Works from a User's Perspective
The system is designed to keep the analysis process simple.

1. **Upload:** First, the user uploads an image using the application. The Main Model interface supports JPG, JPEG, and PNG images and displays a maximum file size of 10 MB.
2. **Preview:** After uploading the image, the application displays a preview along with basic information about the uploaded file.
3. **Analyze:** The user then selects **Analyze Image**.
4. **Display Result:** The system processes the image and displays an authenticity result.

For example, the interface can display:
> **REAL / AUTHENTIC**  
> along with a confidence score.

The application also displays:
- **AI Probability**
- **Real Probability**

This gives the user additional information about the model's prediction.

---

## 6. Main Features

### 6.1 Main Model
The Main Model is the primary model available in the application.

Users can upload an image using:
- **Drag and drop**, or
- **Browse from Computer**.

Once the image is uploaded, the application displays the image preview and file information. The user can then select **Analyze Image** to perform the authenticity analysis. The result is displayed in a dedicated **Authenticity Result** section.

#### **Example Result**
- **Prediction:** `REAL / AUTHENTIC`
- **Confidence Score:** `61.85%`
- **AI Probability:** `38.15%`
- **Real Probability:** `61.85%`

These values provide additional information about the prediction.

### 6.2 Cross-Domain Model
The application also provides a Cross-Domain Model. The interface describes it as a model designed to generalize across different image domains and generation sources.

The user can upload an image and select **Analyze Image**. The result is displayed in the same general format as the Main Model.

#### **Example Result**
- **Prediction:** `REAL / AUTHENTIC`
- **Confidence Score:** `67.87%`
- **AI Probability:** `32.13%`
- **Real Probability:** `67.87%`

---

## 7. Understanding the Prediction
The application provides more information than simply displaying a Real or AI-generated label. Three important values are presented:

1. **Prediction:** Provides the model's classification, such as `REAL / AUTHENTIC`.
2. **Confidence Score:** Indicates the model's confidence in its displayed prediction.
3. **Probability Values:** Provides separate **AI Probability** and **Real Probability** values.

These values allow users to understand how the prediction is distributed between the two categories. However, these values should be interpreted as the model's assessment, rather than absolute proof that an image is genuine or artificially generated.

---

## 8. Grad-CAM Explainability
One of the important features of the application is **Grad-CAM Explainability**.

A prediction from a deep learning model can sometimes be difficult for a user to understand. Grad-CAM provides a visual representation of the areas that contributed more strongly to the model's prediction.

The Grad-CAM page provides:
- **Original Image**
- **Grad-CAM Overlay**
- **Overlay Intensity control**

The user can compare the original image with the visual overlay. The highlighted regions indicate areas where the model had stronger activation during its prediction.

This feature improves the transparency of the system by giving the user a visual indication of where the model focused. The application also appropriately explains that Grad-CAM represents model attention and should not be treated as definitive proof of manipulation or authenticity.

---

## 9. Forensic Analysis Report
The application provides a **Forensic Report** feature. After an image has been analysed, the user can generate a structured report containing information about the analysis.

| Category | Included Details |
| :--- | :--- |
| **Case Information** | Analysis ID, Date and Time, Uploaded Filename |
| **Image Information** | File Type, Resolution, File Size, Color Mode |
| **Model Information** | Model, Model Version, Prediction, Confidence, AI Probability |
| **Preprocessing Pipeline** | Face Detection, Image Resize, Normalization, Tensor Conversion |
| **Explainability Analysis** | Visual heatmaps and model attention metrics |

This makes it possible to retain a structured record of an image authenticity analysis.

---

## 10. Analysis History
The application includes an **Analysis History** page that allows users to review previous analyses.

### **Summary Metrics**
- Total Analyses
- Main Model analyses
- Cross-Domain Model analyses
- Success Rate

### **History Log Table**
Previous analyses are displayed in a structured table containing:
- **Analysis ID**
- **Date and Time**
- **Image**
- **Model**
- **Prediction**
- **Confidence**
- **Status**
- **View Details**

The history feature is useful when multiple images have been analysed and the user wants to review earlier results.

---

## 11. Benefits of the System
The proposed system provides several key benefits:

1. **Simple Interface:** Users can perform image analysis through a graphical interface without directly interacting with programming code.
2. **Automated Analysis:** The system automatically processes the uploaded image and provides an authenticity prediction.
3. **Confidence Information:** The system provides confidence and probability information along with the prediction.
4. **Explainability:** Grad-CAM provides a visual representation of the areas that influenced the model's prediction.
5. **Forensic Documentation:** The forensic report provides a structured record of the analysis.
6. **Analysis History:** Users can review previous analyses through the History section.

---

## 12. Potential Applications
A system for human face authenticity detection can potentially support areas where verification of digital images is important:

- Initial screening of suspicious images.
- Digital content verification.
- Digital forensics.
- Research and academic applications.
- AI-generated content analysis.
- Supporting content moderation processes.

*Note: The system should be considered an analysis and screening tool rather than a replacement for expert forensic investigation.*

---

## 13. Limitations
- **Not Absolute Proof:** The prediction generated by a deep learning system should not be considered absolute proof of authenticity or manipulation.
- **Evolving AI Generators:** AI-based image generation techniques continue to develop, and new generation methods may produce images with characteristics that differ from those encountered during model development.
- **Model Assessment Scope:** The confidence score represents the model's assessment and does not guarantee that the prediction is correct.
- **Attention vs. Manipulation:** Grad-CAM provides information about model attention rather than definitive evidence of manipulation.

Therefore, for high-stakes decisions, the system's output should be considered together with additional evidence and, where appropriate, expert analysis.

---

## 14. Conclusion
**Deep Learning-Based Human Face Authenticity Detection** provides an automated approach for analysing image authenticity through a user-friendly application.

The **Face Forensics – AI Authenticity** application combines authenticity prediction with confidence information, AI and Real probabilities, Grad-CAM explainability, forensic reporting, and analysis history.

The system provides a complete user-facing workflow:

$$	\text{Image Upload} \longrightarrow 	\text{Authenticity Analysis} \longrightarrow 	\text{Prediction} \longrightarrow 	\text{Explanation} \longrightarrow 	\text{Forensic Report} \longrightarrow 	\text{History}$$

By presenting the results through a simple interface, the project makes deep learning-based authenticity analysis more accessible to users without requiring them to understand the underlying technical processes.

---

## Team Declaration

We certify that all team members have actively contributed to the preparation of this document. Each member has reviewed the contents, understands the work presented, and agrees with the submitted report.

**Project:** Deep Learning-Based Human Face Authenticity Detection  
**Team:** Group 11 — Vishakha · Rohit · Aman · Raunak · Somendu  
**Course:** DS & AI Lab Project

| Team Member | Role | Signature |
| --- | --- | --- |
| Vishakha | Pipeline & Presentation Lead | Vishakha |
| Rohit | Training Stability Lead | Rohit |
| Aman | Preprocessing & Transfer Learning Lead | Aman |
| Raunak | Dataset & Bias Analysis Lead | Raunak |
| Somendu | Explainability & Optimisation Lead | Somendu |
