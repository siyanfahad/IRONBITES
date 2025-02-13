# **Anemia Detection Using Nail and Eye Image Samples**  

## **Overview**  
This project utilizes **Convolutional Neural Networks (CNNs)** to detect anemia using images of nails and eyes. By analyzing visual features from these images, the trained deep learning model can classify whether a person has anemia or not.
---

## **Key Features**  
✅ **Deep Learning-Based Anemia Detection** – Uses CNNs for high-accuracy predictions.  
✅ **Dual Input Sources** – Supports both **nail** and **eye** images for improved reliability.  
✅ **Automated Preprocessing** – Normalizes images and extracts relevant features.  
✅ **Fast and Scalable** – Can be deployed as an API for real-time analysis.  
✅ **MERN Stack Integration** – Designed to work with a full-stack web application.  

---

## **Tech Stack**  
- **Machine Learning:** TensorFlow, Keras, OpenCV  
- **Data Processing:** Pandas, NumPy  
- **Frontend & Full-Stack Integration:** MERN Stack (MongoDB, Express.js, React.js, Node.js) -Future Integrations

---

## **How It Works**  

### **1️⃣ Data Collection**  
- The dataset consists of **labeled images of nails and eyes**, categorized into **Anemia** and **Non-Anemia** classes.  

### **2️⃣ Data Preprocessing**  
- Image resizing to **224x224**.  
- Normalization (**pixel values scaled to [0,1]**).  
- Splitting into **training and test sets**.  

### **3️⃣ Model Architecture**  
- **CNN Layers:** Multiple convolutional layers for feature extraction.  
- **Pooling Layers:** Reduces dimensionality while retaining key features.  
- **Dense Layers:** Fully connected layers for classification.  
- **Activation Function:** **Sigmoid** for binary classification.  

### **4️⃣ Training & Evaluation**  
- Trained on **eye and nail datasets** with **binary cross-entropy loss**.  
- **Adam optimizer** for better convergence.  
- **Accuracy & loss tracking** on validation data.  

### **5️⃣ Model Deployment** -Future Enhancement
- The trained model is **saved in Keras format (`.keras`)**.  
- A **Flask API** serves predictions from uploaded images.  - Future Enhancement
- The **MERN stack** integrates this API for real-time detection in a web app.  - Future Enhancement

---

## **Advantages Over Existing Systems**  

| Feature                  | Traditional Systems  | Our CNN-Based Model  |
|--------------------------|---------------------|----------------------|
| **Detection Approach**   | Manual blood tests  | AI-based image analysis |
| **Invasiveness**         | Requires blood sample | Non-invasive (uses images) |
| **Speed**               | Slow (lab tests needed) | Instant results |
| **Accessibility**       | Requires medical facilities | Can be used remotely |
| **Scalability**         | Limited | Easily deployable as a web service |

---

