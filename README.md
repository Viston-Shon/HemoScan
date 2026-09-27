# HemoScan
An automated digital image processing system that leverages foundation vision models to screen blood smears and nail images for cell morphology and classification.


An automated digital image processing and computer vision system for peripheral blood smear analysis. HemoScan screens for red blood cell (RBC) morphology and classification using traditional image processing techniques combined with a deep learning foundation model.

## 🚀 Features
* **Advanced Preprocessing:** Giemsa-aware background removal, non-local means denoising, and CLAHE contrast enhancement.
* **Robust Segmentation:** Local maxima watershed segmentation with dynamic morphological closing to separate overlapping cells.
* **AI-Powered Classification:** MobileNetV3Large model classifies isolated cells into 10 distinct morphological categories.
* **Clinical Reporting:** Automatically calculates cell type distributions and generates diagnostic recommendations.

## 📁 Project Structure
* `/frontend` - Web interface (`quickAneFrontend.html`, `quickAneLogin.html`)
* `/backend` - Flask server and image processing pipeline (`quickAneBackend.py`, `requirements.txt`)
* `/models` - Pre-trained `.h5` weights and class mapping JSON

## 🛠️ Steps to Run

### 1. Start the Backend
The backend processes the images and runs the AI model using Python and Flask.

1. Open your terminal and navigate to the root of the `HemoScan` folder:
   cd path/to/HemoScan

2. Create a virtual environment so dependencies don't conflict with your system:
   python -m venv .venv

3. Activate the virtual environment:
   * Windows: .venv\Scripts\activate
   * Mac/Linux: source .venv/bin/activate

4. Install the required libraries:
   pip install -r backend/requirements.txt

5. Run the server:
   python backend/quickAneBackend.py
   
   (Keep this terminal open. The backend is now listening at http://127.0.0.1:5000)

### 2. Launch the Frontend
The frontend is built with static HTML and JavaScript and communicates with the Python server you just started.

1. Open the `HemoScan` folder in your file explorer.
2. Navigate into the `frontend` folder.
3. Double-click `quickAneLogin.html` to open it in your default web browser (Chrome, Edge, Safari).
