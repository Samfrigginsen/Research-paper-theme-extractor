\boxed{
```markdown
# Theme Extractor GUI Application

A PyQt6-based tool for analyzing research papers to extract themes using keyword matching (single file) and LDA topic modeling (multiple files).

## Features
- Single file analysis with keyword matching
- Multi-file analysis using LDA topic modeling
- PDF text extraction with multiple fallback methods
- Progress tracking and result export to Excel
- CSV input support for pre-processed text

## Installation
1. Clone repository:
```bash
git clone [your-repository-url]
cd theme-extractor-gui
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Requirements
- Python 3.7+
- PyQt6
- pandas
- PyPDF2
- spaCy
- gensim
- python-dateutil

## Usage
### From Source
```bash
python theme_extractor_gui.py
```

### As Executable
1. Build with PyInstaller:
```bash
pyinstaller --onefile --windowed theme_extractor_gui.py
```

2. Run from `dist` folder:
```bash
./dist/theme_extractor_gui.exe
```

## Features
- Single File Mode: 
  - Upload individual PDFs
  - Keyword-based theme matching
- Multi-File Mode:
  - Process folders of PDFs
  - Automatic theme detection (LDA)
  - Adjustable number of themes
- Results:
  - In-app display
  - Excel export with timestamps
  - Coherence scores for LDA models

## Project Structure
- `theme_extractor_gui.py`: Main application window and GUI logic
- `theme_extractor.py`: Core text processing and analysis module
  - Preprocessing pipeline
  - Theme definitions (THEMES dictionary)
  - LDA model handling
