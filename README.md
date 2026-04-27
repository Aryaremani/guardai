  GuardAI - Automated Content Moderation System

GuardAI is a real-time Machine Learning application designed to detect and flag toxic user-generated content online. It classifies text across 6 different harm categories using Natural Language Processing (NLP) and robust Machine Learning algorithms.

###  Live Demo
** [Try GuardAI Live on Vercel](https://guardai-pqwd2qw0f-aryaremanis-projects.vercel.app)** *(Note: Use the exact URL assigned to your Vercel deployment if this changes)*

---

##  Features
- **Real-Time Analysis**: Type or paste any text and receive instant toxicity scores.
- **Multi-Label Classification**: Detects overlapping categories (e.g., something can be both "Obscene" and a "Threat").
- 6 Moderation Categories Supported:
  -  Toxic
  -  Severe Toxic
  -  Obscene
  -  Threat
  -  Insult
  -  Identity Hate
- **Lightweight & Fast**: Built with efficient TF-IDF extraction and Logistic Regression to run in Serverless environments with extreme low-latency.

##  System Architecture

* **Frontend**: HTML5, CSS3, JavaScript (Glassmorphism UI, Responsive Design)
* **Backend**: Python 3.11 Serverless API (Hosted on Vercel Functions)
* **Machine Learning**: `scikit-learn` (Logistic Regression using the One-Vs-Rest strategy)
* **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
* **Model Serialization**: `joblib`

##  Running Locally

### 1. Requirements
Make sure you have Python 3.11 installed.

### 2. Setup
Clone the repository and install the exact Python dependencies used during training:
```bash
git clone https://github.com/Aryaremani/guardai.git
cd guardai
pip install -r requirements.txt
```

### 3. Run the Backend API natively
To test the inference locally without a big server, you can use the smoke test:
```bash
python smoke_test.py
```
*(Alternatively, you can run a local HTTP server inside the `api/` or `public/` directories).*

### 4. Vercel Dev Server 
The best way to run both the frontend and the Python serverless API locally is via the Vercel CLI:
```bash
npm i -g vercel
vercel dev
```
This will spin up a `localhost:3000` instance simulating the exact Vercel Cloud environment.

##  Training Pipeline
If you wish to re-train the model on new data:
1. Ensure you have the Wikipedia Toxic Comments `.csv` files stored in `/data/`
2. Run the preprocessing script: `python src/preprocess.py`
3. Run the train script: `python src/train.py`
4. The new `.joblib` model weights will automatically be saved to `/models/`.

##  License
This project is open-source. Feel free to use and modify the code.
