# Deep Learning-Based Human Face Authenticity Detection
## User Guide for Face Forensics – AI Authenticity

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Purpose of the Application](#2-purpose-of-the-application)
3. [Application Overview](#3-application-overview)
4. [Getting Started](#4-getting-started)
5. [Main Model](#5-main-model)
6. [Uploading an Image](#6-uploading-an-image)
7. [Running an Analysis](#7-running-an-analysis)
8. [Understanding the Analysis Result](#8-understanding-the-analysis-result)
9. [Understanding Prediction and Probability](#9-understanding-prediction-and-probability)
10. [Cross-Domain Model](#10-cross-domain-model)
11. [Grad-CAM Explainability](#11-grad-cam-explainability)
12. [Understanding the Grad-CAM Visualization](#12-understanding-the-grad-cam-visualization)
13. [Forensic Analysis Report](#13-forensic-analysis-report)
14. [Information Included in the Forensic Report](#14-information-included-in-the-forensic-report)
15. [Downloading the Forensic Report](#15-downloading-the-forensic-report)
16. [Analysis History](#16-analysis-history)
17. [History Summary](#17-history-summary)
18. [Searching Analysis History](#18-searching-analysis-history)
19. [Filtering Analysis History](#19-filtering-analysis-history)
20. [Viewing Previous Analysis Details](#20-viewing-previous-analysis-details)
21. [Built-in User Guide](#21-built-in-user-guide)
22. [Understanding Confidence Scores](#22-understanding-confidence-scores)
23. [How to Interpret a Result](#23-how-to-interpret-a-result)
24. [Important Limitations](#24-important-limitations)
25. [Recommended Usage](#25-recommended-usage)
26. [Complete User Workflow](#26-complete-user-workflow)
27. [Quick Reference Guide](#27-quick-reference-guide)
28. [Common User Errors](#28-common-user-errors)
29. [Conclusion](#29-conclusion)

---

## 1. Introduction
**Deep Learning-Based Human Face Authenticity Detection** is a deep-learning-based application designed to assist users in analysing whether a human face image is **Real / Authentic** or **AI Generated**.

The application is presented through a user-friendly interface called **Face Forensics – AI Authenticity**. 

The system allows a user to upload an image, select an analysis model, obtain an authenticity prediction, view confidence and probability information, inspect a Grad-CAM explanation, generate a forensic report, and review previous analyses. This document provides step-by-step instructions for using the application.

---

## 2. Purpose of the Application
The purpose of the application is to provide users with an automated assessment of the authenticity of human face images. Instead of manually inspecting an image, the user can upload the image to the application and allow the trained model to analyse it.

The application provides more than just a final prediction. It also provides supporting information such as:
- Authenticity prediction
- Confidence score
- AI probability
- Real probability
- Model information
- Grad-CAM visualization
- Forensic report
- Analysis history

These features allow users to understand and document the result of an analysis.

---

## 3. Application Overview
The application contains the following major sections:
- **Main Model**
- **Cross-Domain Model**
- **Grad-CAM**
- **Forensic Report**
- **History**
- **User Guide**

The application also provides a **Light/Dark mode** option.

### General Workflow
$$	\text{Select Model} \longrightarrow 	\text{Upload Image} \longrightarrow 	\text{Click Analyze} \longrightarrow 	\text{Review Result} \longrightarrow 	\text{Open Grad-CAM} \longrightarrow 	\text{Generate Report}$$

A new user can perform an image analysis without needing to interact directly with the underlying deep-learning code.

---

## 4. Getting Started

### 4.1 Using the live application (recommended)

Open the deployed app in any web browser:

**https://group-11-ds-and-ai-lab-project.onrender.com**

No installation is required. On the free hosting tier, the first visit
after a period of inactivity may take **30–60 seconds** while the server
starts — this is normal. Refresh if the page seems stuck.

The live site includes **Main Model** and **Cross-Domain Model** pages.
Advanced pages (Manipulation Robustness, Model Comparison) require extra
model files and are only available when running the app locally — see
`../docs/READMEdeployment.md`.

### 4.2 Navigating the application

When the application is opened, the main navigation menu is available on the left side. The navigation menu allows the user to move between the different sections of the application.

The application provides two main choices for image analysis:
1. **Main Model**
2. **Cross-Domain Model**

A user should select the model according to the type of analysis they want to perform.

The application also indicates its system status. When the system is ready, the interface displays:
> **System Ready – All systems operational**

---

## 5. Main Model
The **Main Model** is the primary model available in the application for face authenticity analysis.

### Steps to Use the Main Model
1. Select **Main Model** from the navigation menu.
2. Upload an image.
3. Check the uploaded image.
4. Click **Analyze Image**.
5. Wait for the analysis to complete.
6. Review the result.

#### **Figure 1. Main Model Interface**
![Figure 1: Main Model Interface](Images/main_model.png)
The Main Model interface contains an area for uploading an image and displaying the selected image before analysis. The user can select an image from the computer using the **Browse from Computer** option.

---

## 6. Uploading an Image
To upload an image:
- **Step 1:** Open the **Main Model** page.
- **Step 2:** Click **Browse from Computer**.
- **Step 3:** Select the required image from your computer.

The interface also provides an upload area where the user can drag and drop an image.

### Supported File Constraints
- **Supported Formats:** `JPG`, `JPEG`, `PNG`
- **Maximum File Size:** `10 MB`

#### **Figure 2. Image Upload Interface**
![Figure 2: Image Upload Interface](Images/main_model1.png)

After the image is uploaded, the application displays a preview and information about the selected file. The information includes:
- **File Type:** `JPG`
- **Resolution:** `512 × 512`
- **File Size:** `27.7 KB`

The user should check the preview before starting the analysis.

---

## 7. Running an Analysis
After uploading the image, click:
> **Analyze Image**

The application will process the uploaded image. The user should wait for the analysis to complete before leaving the page or starting another analysis.

If the wrong image has been selected, the user can click **Replace Image** and select another image. Once processing is complete, the application displays the **Authenticity Result**.

---

## 8. Understanding the Analysis Result
The completed analysis provides several pieces of information.

#### **Figure 3. Completed Main Model Analysis**
![Figure 3: Completed Main Model Analysis](Images/main_model1.png)

The result screen contains:
- **Prediction:** `REAL / AUTHENTIC`
- **Confidence:** `61.85%`
- **AI Probability:** `38.15%`
- **Real Probability:** `61.85%`
- **Model:** `Main Model`
- **Model Version:** `v1.0`
- **Analysis Status:** `Completed`
- **Analysis ID:** `FA-000126`

### Analysis ID
The Analysis ID is a unique identifier associated with a particular analysis. It is useful when locating the analysis later in the **History** section.

---

## 9. Understanding Prediction and Probability
The application presents the result using several complementary values:

- **Prediction:** The classification produced by the model (e.g., `REAL / AUTHENTIC`).
- **AI Probability:** The probability value displayed by the application for the AI-generated class (e.g., `38.15%`).
- **Real Probability:** The probability value displayed for the real/authentic class (e.g., `61.85%`).
- **Confidence:** Represents how strongly the model supports its displayed prediction.

> These values should be considered together rather than looking at only one value.

---

## 10. Cross-Domain Model
The application provides a second model called the **Cross-Domain Model**. The application's built-in User Guide describes this model as being designed to generalize across:
- Different image domains
- Different cameras
- Different generation sources

### How to use the Cross-Domain Model
1. Select **Cross-Domain Model** from the navigation menu.
2. Select **Browse from Computer**.
3. Choose the required image.
4. Check the image preview.
5. Click **Analyze Image**.
6. Wait for the analysis to complete.
7. Review the displayed result.

#### **Figure 4. Cross-Domain Model Interface**
![Figure 4: Cross-Domain Model Interface](Images/cross_domain.png)

The Cross-Domain Model follows the same general upload and analysis process as the Main Model. The result includes the prediction and supporting information such as confidence (`67.87%`) and probability values (`32.13%` AI vs. `67.87%` Real).

---

## 11. Grad-CAM Explainability
The application provides a **Grad-CAM** feature to help users understand the model's visual attention during prediction.

### How to Access Grad-CAM
1. Complete an image analysis.
2. Open the **Grad-CAM** section (or click **View Grad-CAM**).
3. Select the relevant analysis if required.
4. Review the original image and Grad-CAM visualization.

#### **Figure 5. Grad-CAM Interface**
![Figure 5: Grad-CAM Interface](Images/Grad_Cam.png)

The Grad-CAM page provides information such as:
- **Analysis ID** (`FA-000127`)
- **Model** (`Cross-Domain Model`)
- **Prediction** (`Real`)
- **Confidence** (`67.87%`)
- **Original Image** vs. **Grad-CAM Overlay**

The application also provides an **Overlay Intensity** control (slider) that users can adjust to change the transparency/intensity of the heatmap visualization.

---

## 12. Understanding the Grad-CAM Visualization
The Grad-CAM visualization helps the user understand which regions received stronger activation from the model.

- **Red/Yellow areas:** Represent higher activation.
- **Blue areas:** Represent lower activation.

The interface provides different viewing options:
- `Original`
- `Heatmap`
- `Overlay`

> **Important:** Grad-CAM does not prove that a particular region is manipulated. It represents the model's attention during its prediction. Therefore, the user should interpret the Grad-CAM image as an explanation of model behaviour rather than definitive forensic evidence.

---

## 13. Forensic Analysis Report
The application provides a **Forensic Report** feature. The report provides structured information about an analysis and allows the user to document the result.

### To Generate the Report
1. Complete an image analysis.
2. Open the **Forensic Report** section or select **Generate Forensic Report**.
3. Review the displayed information.
4. Use the available **Download Report** option if a copy is required.

#### **Figure 6. Forensic Report**
![Figure 6: Forensic Report](Images/Forensic_Analysis.png)

---

## 14. Information Included in the Forensic Report
The report is divided into several clear sections:

### 14.1 Case Information
Identifies the individual analysis:
- **Analysis ID** (`FA-000127`)
- **Date and Time** (`August 11, 2026, 02:35 PM`)
- **Uploaded Filename** (`2.jpg`)

### 14.2 Image Information
Provides technical details about the input image:
- **File Type** (`JPEG`)
- **Resolution** (`600 × 902`)
- **File Size** (`153.4 KB`)
- **Color Mode** (`RGB`)

### 14.3 Model Information
Records the model and results:
- **Model** (`Cross-Domain Model`)
- **Model Version** (`v1.0`)
- **Prediction** (`REAL / AUTHENTIC`)
- **Confidence** (`67.87%`)
- **AI Probability** (`32.13%`)

### 14.4 Inference Preprocessing Pipeline
Displays preprocessing performed before inference:
- **Face Detection** (`Applied`)
- **Image Resize** (`Applied`)
- **Normalization** (`Applied`)
- **Tensor Conversion** (`Applied`)

### 14.5 Explainability Analysis
Contains the explainability information associated with the analysis, including visual heatmaps.

---

## 15. Downloading the Forensic Report
After reviewing the forensic report, the user can select:
> **Download Report**

This allows the user to retain a copy of the analysis information for record keeping or later review.

---

## 16. Analysis History
The application provides an **Analysis History** page allowing users to view previously performed analyses.

#### **Figure 7. Analysis History**
![Figure 7: Analysis History](Images/History.png)

The History page displays information such as Analysis ID, Date & Time, Image thumbnail, Model, Prediction, Confidence, and Status.

---

## 17. History Summary
The History page provides summary metrics at the top:
- **Total Analyses** (e.g., `4`)
- **Main Model Analyses** (e.g., `3`)
- **Cross-Domain Model Analyses** (e.g., `1`)
- **Success Rate** (e.g., `100.0%`)

---

## 18. Searching Analysis History
A user can search for a previous analysis using:
- **Analysis ID:** Enter the relevant ID in the search field (e.g., `FA-000126`).
- **Filename:** Enter the filename associated with the analysed image (e.g., `3.jpg`).

---

## 19. Filtering Analysis History
Users can filter previous analyses based on:
- **Model:** Main Model / Cross-Domain Model
- **Result:** Authentic / AI Generated

To remove filters, select **Clear Filters**.

---

## 20. Viewing Previous Analysis Details
To view a previous analysis:
1. Open **History**.
2. Find the required analysis using search or filters.
3. Select **View Details**.

This avoids the need to repeat an analysis simply to access previously generated information.

---

## 21. Built-in User Guide
The application contains a built-in **User Guide** section accessible directly from the sidebar menu.

#### **Figure 8. Application User Guide**
![Figure 8: Application User Guide](Images/User_Guide.png)

The built-in User Guide explains:
- What the system does
- Model differences
- Analysis workflow
- Confidence interpretation
- Grad-CAM guidance
- Important limitations

---

## 22. Understanding Confidence Scores
The application's User Guide provides the following interpretation of confidence scores:

| Confidence Score | Interpretation |
| :--- | :--- |
| **90% and above** | Very High Confidence |
| **70% – 90%** | High Confidence |
| **50% – 70%** | Moderate Confidence |
| **Below 50%** | Low Confidence |

*For example, a confidence score of 61.85% falls into the **Moderate Confidence** range.*

> **Key Rule:** Confidence reflects model certainty, not absolute truth.

---

## 23. How to Interpret a Result
When reviewing an analysis, consider all parameters together:
1. **Prediction:** Check whether classified as Real/Authentic or AI Generated.
2. **Confidence:** Check how strongly the model supports its prediction.
3. **AI Probability:** Check the calculated AI probability.
4. **Real Probability:** Check the calculated Real probability.
5. **Grad-CAM:** Examine visual attention regions if needed.

---

## 24. Important Limitations
- A model prediction is **not absolute proof**.
- A high confidence score does not guarantee correctness.
- A low confidence score indicates greater uncertainty.
- Grad-CAM shows **model attention**, not proof of manipulation.
- The system should **not replace professional forensic investigation**.

---

## 25. Recommended Usage
- **Use appropriate images:** Clear face images in JPG, JPEG, or PNG formats.
- **Verify before analysis:** Check preview, filename, resolution, and file size.
- **Review all metrics:** Do not rely solely on the final classification label.
- **Use Grad-CAM when needed:** For visual validation of model focus.
- **Generate reports:** Save documentation for official record keeping.
- **Keep track of IDs:** Save Analysis IDs for easy retrieval.

---

## 26. Complete User Workflow

$$	\text{Step 1: Open Application} \longrightarrow 	\text{Step 2: Select Model} \longrightarrow 	\text{Step 3: Upload Image} \longrightarrow 	\text{Step 4: Verify Preview}$$
$$\downarrow$$
$$	\text{Step 8: Generate Report} \longleftarrow 	\text{Step 7: Examine Grad-CAM} \longleftarrow 	\text{Step 6: Review Result} \longleftarrow 	\text{Step 5: Click Analyze}$$
$$\downarrow$$
$$	\text{Step 9: Review History Later}$$

---

## 27. Quick Reference Guide

| Function | Where to Find It | What It Does |
| :--- | :--- | :--- |
| **Main Analysis** | Main Model | Performs primary authenticity analysis |
| **Cross-Domain Analysis** | Cross-Domain Model | Performs cross-domain analysis |
| **Upload Image** | Browse from Computer | Selects an image for analysis |
| **Start Analysis** | Analyze Image | Starts model inference |
| **Change Image** | Replace Image | Uploads another image |
| **View Prediction** | Authenticity Result | Displays classification label |
| **View Confidence** | Result Section | Displays model confidence percentage |
| **View Probabilities** | Result Section | Displays AI and Real probabilities |
| **Explainability** | Grad-CAM | Displays visual model attention |
| **Report** | Forensic Report | Displays detailed analysis report |
| **Download Report** | Download Report | Saves report to local storage |
| **Previous Analyses** | History | Displays earlier analysis logs |
| **Search/Filter** | History | Filters logs by ID, name, or model |
| **Instructions** | User Guide | Explains application usage |
| **Appearance** | Light/Dark Toggle | Changes UI color theme |

---

## 28. Common User Errors

- **Error 1: Uploading an unsupported file**
  - *Solution:* Convert file to JPG, JPEG, or PNG.
- **Error 2: File exceeds size limit**
  - *Solution:* Compress image below 10 MB.
- **Error 3: Wrong image selected**
  - *Solution:* Click **Replace Image**.
- **Error 4: Unable to find an old analysis**
  - *Solution:* Use search filters in the **History** tab.
- **Error 5: Misinterpreting Grad-CAM**
  - *Solution:* Remember Grad-CAM indicates focus, not guaranteed manipulation.

---



## 29. Conclusion
The **Face Forensics – AI Authenticity** application provides a comprehensive, accessible interface for deep-learning-based human face authenticity analysis. 

By combining dual detection models, confidence metrics, Grad-CAM explainability, forensic reporting, and historical tracking, the application simplifies complex AI evaluations into an intuitive workflow. Users must always treat results as analytical assessments to complement expert forensic evaluation.

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
