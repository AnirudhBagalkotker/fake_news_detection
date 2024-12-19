# **Fake News Detection Using Hybrid CNN-LSTM Model**

## **Overview**

This project implements a hybrid deep learning model combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to detect fake news. The system leverages the strengths of both architectures to extract spatial and sequential features, achieving a high accuracy of 99.05%. The project includes a comprehensive pipeline for data preprocessing, model training, evaluation, and deployment using Dockerized services for both the backend and frontend.

---

## **Features**

- Automated detection of fake news using a hybrid CNN-LSTM model.
- Preprocessing pipeline with tokenization, lemmatization, and GloVe word embeddings.
- Evaluation metrics such as accuracy, precision, recall, F1 score, and confusion matrix.
- Dockerized architecture for seamless deployment:
  - **Backend:** Model API service for predictions.
  - **Frontend:** Web-based user interface for interacting with the system.

---

## **Project Structure**

The repository is organized as follows:

```
.
├── backend/
│   ├── datasets/             # True, Fake, and Saved Processed datasets
|   |-- models/               # Saved model weights and checkpoints
│   ├── main.py               # FastAPI for model predictions
│   ├── model.py              # CNN-LSTM hybrid model
│   ├── model.ipynb           # Jupyter notebook for model development
│   ├── importModel.py        # Import model definitions for main.py
│   ├── vocab.json            # Saved word to index mapping
│   ├── glove_embeddings.npy  # Saved GloVe word embeddings 
│   ├── requirements.txt      # Backend dependencies
├── frontend/
│   ├── src/                  # Frontend application files (App.js, App.css, etc.)
│   ├── public/               # Static assets
|   |-- build/                # Built frontend assets
│   ├── package.json          # Frontend dependencies
├── reports/                  # Generated reports, figures, and tables
├── Dockerfile.backend        # Dockerfile for backend service
├── Dockerfile.frontend       # Dockerfile for frontend service
├── docker-compose.yml        # Docker Compose configuration
├── README.md                 # Project documentation
```

---

## **Getting Started**

### **Prerequisites**

- Python 3.12
- Docker and Docker Compose installed

### **Setup**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/username/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Download files from Drive**

   Download files from the link:
   [https://drive.google.com/drive/folders/1XnD0qOXhAmWQqA8xJ6RYXzgXU-Lkf-Ng](https://drive.google.com/drive/folders/1XnD0qOXhAmWQqA8xJ6RYXzgXU-Lkf-Ng)  and place it in the /backend directory (NOTE: Dont Place it the /backend/models directory)

3. **Build Docker Containers**Use Docker Compose to build and run the backend and frontend services:

   ```bash
   docker-compose build
   docker-compose up
   ```

4. **Access the Application**
   Once the services are running, open your browser and navigate to:
   `http://localhost:8000` for the interface.

## OR

1. **Clone the Repository**

   ```bash
   git clone https://github.com/username/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Install Dependencies for Backend (using only Python 3.12)**

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Install Dependencies for Frontend and Build**

   ```bash
   cd frontend
   npm install
   npm run build
   ```

4. **Download files from Drive (Optional)**

   Download files from the link:
   [https://drive.google.com/drive/folders/1XnD0qOXhAmWQqA8xJ6RYXzgXU-Lkf-Ng](https://drive.google.com/drive/folders/1XnD0qOXhAmWQqA8xJ6RYXzgXU-Lkf-Ng)  and place it in the /backend directory (NOTE: Dont Place it the /backend/models directory)

5. **Run the Model (model.ipynb)**

   You can skip this step if you download the checkpoints from the drive.
6. **Run the Backend**

   Run the backend after model generates the hybrid checkpoint. You can skip this step if you download the checkpoints from the drive.

   ```bash
   cd backend
   uvicorn main:app --reload
   ```

7. **Access the Application**
   Once the services are running, open your browser and navigate to:
   `http://localhost:8000` for the interface.

---

## **How It Works**

### **Backend**

The backend is a FastAPI-based API that:

- Loads the trained CNN-LSTM hybrid model.
- Accepts input data (news article text or URL) via API requests.
- Returns predictions (True or Fake) in JSON format.

### **Frontend**

The frontend is a web-based interface built with React. It allows users to:

- Input news articles for analysis.
- View results and model confidence scores.

---

## **Usage**

1. **Upload a News Article**Input a news article's text or URL into the frontend.
2. **Get Predictions**
   The system processes the text and displays the prediction (True or Fake).

---

## **Results**

- **Accuracy:** 99.05%
- **Precision:** 99.06%
- **Recall:** 99.05%
- **F1 Score:** 99.05%

For detailed results and additional figures, refer to the **reports/** directory.

---

## **Development**

### **Backend Development**

Run the FastAPI app locally:

```bash
cd backend
uvicorn main:app --reload
```

### **Frontend Development**

Run the frontend locally:

```bash
cd frontend
npm install
npm run build
```

---

## **Technologies Used**

- **Deep Learning Frameworks:** PyTorch, Sklearn, numpy, pandas
- **Text Processing:** NLTK, gensim,
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Backend Framework:** FastAPI
- **Frontend Framework:** React
- **Containerization:** Docker, Docker Compose

---

## **Contributing**

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to modify.

---

## **Acknowledgments**

This project builds on prior research and openly available datasets for fake news detection.
