# FMNarrative
AI-driven Football Manager 24 press conference assistant aimed at improving immersion - realistic questions, dynamic answers, context-aware tone.

## Installation

### 1. System Dependencies (C++ libraries)

**macOS (Homebrew):**
```bash
brew install cmake opencv tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install cmake libopencv-dev tesseract-ocr libtesseract-dev libleptonica-dev
```

### 2. Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build C++ OCR Component
```bash
mkdir build && cd build
cmake ..
make
cd ..
```

### 4. Environment Setup
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_api_key_here
```
