```markdown
# Research Paper Theme Extractor (GUI)

A PyQt6-based application for analyzing research papers to extract themes using:
- **Keyword matching** for single PDF analysis
- **LDA topic modeling** for multi-file analysis

## Features
- **Dual Analysis Modes**:
  - Single PDF analysis with keyword matching
  - Multi-file analysis with LDA topic modeling
- **Flexible Input**:
  - Process individual PDFs
  - Analyze folders of PDFs
  - Use pre-processed CSV files
- **Smart Output**:
  - In-app results display
  - Excel exports with timestamps
  - Coherence score reporting
- **Robust PDF Handling**:
  - Multiple text extraction fallback methods
  - Error logging and recovery

## Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/theme-extractor-gui.git
cd theme-extractor-gui
```

2. **Set Up Virtual Environment**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

1. **Launch Application**
```bash
python theme_extractor_gui.py
```

2. **Choose Analysis Mode**:
   - 🗂️ **Folder Mode**: Analyze multiple PDFs
     1. Click "Analyze Folder"
     2. Select PDF folder
     3. Set number of themes (default: 12)
   - 📄 **Single File Mode**: Analyze individual PDFs
     1. Click "Analyze File"
     2. Select PDF file

3. **View Results**:
   - Results display in application
   - Excel file saved to source folder:
     `extracted_themes_YYYYMMDD_HHMMSS.xlsx`


## Requirements
- Python 3.7+
- PyQt6 >= 6.0
- pandas >= 1.3
- PyPDF2 >= 2.0
- spaCy >= 3.0
- gensim >= 4.0


## Notes
- Default LDA topics: 12 (adjust via number spinner)
- Processing time varies by paper count/length
- Includes multiple PDF text extraction fallback methods
- Coherence score shown for LDA model validation
