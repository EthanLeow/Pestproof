# 🪲 Agricultural Pest Identification System

An AI-powered web application that helps farmers identify agricultural pests and receive organic treatment recommendations. The system uses deep learning for image classification and a local LLM for generating contextual advice.

## 📋 Project Overview

This project combines computer vision and natural language processing to:
- Identify 12 different types of agricultural pests from images
- Provide crop-specific pest analysis
- Generate organic treatment and prevention recommendations
- Classify insects as beneficial, harmful, or neutral to specific crops

### Supported Pest Classes

- Ants
- Bees
- Beetle
- Caterpillar
- Earthworms
- Earwig
- Grasshopper
- Moth
- Slug
- Snail
- Wasp
- Weevil

### Supported Crops

- Tomato
- Corn
- Wheat
- Rice
- Soybean

## 🏗️ Architecture

### Backend (FastAPI + TensorFlow)
- **Model**: EfficientNetV2-B0 (transfer learning)
- **Accuracy**: 90-95% on validation set
- **Model Size**: ~24MB (mobile-friendly)
- **Image Processing**: 224x224 RGB images
- **API Framework**: FastAPI with CORS support

### Frontend (React)
- **UI Library**: React 18
- **Styling**: Custom CSS with gradient backgrounds
- **Markdown Rendering**: ReactMarkdown for LLM responses
- **File Handling**: Native file input with image preview

### LLM Integration
- **Local LLM**: DeepSeek-R1-0528-Qwen3-8B via LM Studio
- **Purpose**: Generate organic farming advice and treatment plans
- **Response Format**: Structured recommendations with crop context

## 📊 Model Performance

Based on [`OIP_Grp2.ipynb`](backend/OIP_Grp2.ipynb):

```
Training Accuracy: 94.7%
Validation Accuracy: 90.2%
Test Accuracy: 92.5%

Best Performing Classes:
- Moth: 100% accuracy
- Snail: 100% accuracy
- Ants: 98.7% accuracy

Deployment Status: APPROVED - Ready for deployment
```

### Data Split
- **Training**: 70% (3,839 samples)
- **Validation**: 15% (823 samples)
- **Test**: 15% (823 samples)

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 16+
- LM Studio (for local LLM)
- 8GB RAM minimum (16GB recommended)

### Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd OIP
```

#### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

Required packages (from [`backend/requirements.txt`](backend/requirements.txt)):
- fastapi
- uvicorn
- tensorflow
- pillow
- numpy
- python-multipart
- requests

#### 3. Frontend Setup

```bash
cd client
npm install
```

Required packages (from [`client/package.json`](client/package.json)):
- react
- react-dom
- react-markdown

#### 4. LM Studio Setup

1. Download and install LM Studio from https://lmstudio.ai
2. Download the model "DeepSeek-R1-0528-Qwen3-8B" in LM Studio
3. Enable the LM Studio Local Server:
   - Open LM Studio
   - Go to Settings > Developer
   - Enable "Local Server"
   - Ensure it's running on `http://localhost:1234`

### Running the Application

#### Start the Backend

```bash
cd backend
uvicorn backend:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

#### Start the Frontend

Open a new terminal:

```bash
cd client
npm start
```

The frontend will be available at `http://localhost:3000`

## 💻 Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Select an image of an insect/pest from your device
3. Choose the crop type from the dropdown menu
4. Click "🚀 Upload & Identify"
5. View the results:
   - Predicted insect class
   - Confidence score
   - LLM-generated advice (crop context, diagnosis, treatment, prevention)

## 📁 Project Structure

```
OIP/
├── backend/
│   ├── backend.py              # FastAPI server and prediction endpoint
│   ├── best_model.keras        # Trained EfficientNetV2 model
│   ├── requirements.txt        # Python dependencies
│   ├── OIP_Grp2.ipynb         # Main training notebook (EfficientNetV2)
│   ├── selfbuildmodel.ipynb   # Custom CNN training notebook
│   └── EffNet_L.ipynb         # EfficientNetV2-L experiments
│
├── client/
│   ├── src/
│   │   ├── App.jsx            # Main React component
│   │   ├── App.css            # Application styling
│   │   └── index.js           # React entry point
│   ├── public/
│   │   └── index.html         # HTML template
│   └── package.json           # Node dependencies
│
└── README.md
```

## 🔧 API Reference

### POST `/predict`

Predict pest class from an uploaded image.

**Request:**
- `file`: Image file (multipart/form-data)
- `crop`: Crop type (form parameter)

**Response:**
```json
{
  "predicted_class": "beetle",
  "confidence": 0.95,
  "llm_response": "Crop Context: Insect is harmful to tomato.\n\nDiagnosis: Beetles can damage leaves...\n\nTreatment: Apply neem oil spray...\n\nPrevention: Practice crop rotation..."
}
```

## 🧪 Model Training

The model was trained using the following approach (see [`OIP_Grp2.ipynb`](backend/OIP_Grp2.ipynb)):

### Training Configuration
- **Base Model**: EfficientNetV2-B0 (pre-trained on ImageNet)
- **Transfer Learning**: Frozen base, custom classification head
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 8 (CPU-optimized)
- **Epochs**: 4 with early stopping

### Data Augmentation (Training Only)
- Rotation: ±20°
- Width/Height Shift: 20%
- Zoom: 20%
- Horizontal Flip: Enabled

### Model Architecture
```
EfficientNetV2-B0 (frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, activation='relu')
    ↓
Dense(12, activation='softmax')
```

### Training Results (from [`OIP_Grp2.ipynb`](backend/OIP_Grp2.ipynb))
```
Epoch 4/4
Training Accuracy: 94.7%
Validation Accuracy: 90.2%
Training completed in ~2 hours (CPU)
```

## 📝 Alternative Model Approaches

The repository includes two additional training notebooks:

1. **[`selfbuildmodel.ipynb`](backend/selfbuildmodel.ipynb)**: Custom CNN architecture
   - 6 convolutional layers with batch normalization
   - 22M parameters
   - Useful for understanding CNN fundamentals

2. **[`EffNet_L.ipynb`](backend/EffNet_L.ipynb)**: EfficientNetV2-Large experiments
   - Larger model variant
   - Higher capacity but slower inference

## 🐛 Troubleshooting

### LM Studio Connection Issues
- Ensure LM Studio is running and the local server is enabled
- Check that the endpoint is `http://localhost:1234/v1/chat/completions`
- Verify the model name in [`backend.py`](backend/backend.py) matches your loaded model

### Model Loading Errors
- Ensure `best_model.keras` is in the [`backend`](backend) directory
- Check TensorFlow version compatibility (2.19.0+ recommended)

### CORS Errors
- The backend allows all origins by default (see [`backend.py`](backend/backend.py))
- For production, update `allow_origins` to specific domains

### Low Prediction Accuracy
- Ensure images are clear and well-lit
- Image should focus on a single insect
- Supported resolution: 224x224 pixels (automatically resized)

## 📊 Performance Metrics

From [`OIP_Grp2.ipynb`](backend/OIP_Grp2.ipynb):

```
Per-Class Accuracy:
  ants: 98.7%
  bees: 96.0%
  beetle: 74.6%
  catterpillar: 76.9%
  earthworms: 77.6%
  earwig: 75.4%
  grasshopper: 88.9%
  moth: 100.0%
  slug: 93.1%
  snail: 100.0%
  wasp: 96.0%
  weevil: 98.6%
```

## 🎯 Future Improvements

- [ ] Add more pest classes
- [ ] Support mobile app deployment
- [ ] Implement batch image processing
- [ ] Add multi-language support
- [ ] Include pest population density estimation
- [ ] Real-time video pest detection

## 📄 License

This project is for educational and agricultural use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This project uses CPU-optimized training. For faster training with GPU, adjust batch sizes and install tensorflow-gpu.