# Research Paper Theme Extractor

This script extracts dominant themes from research papers using topic modeling techniques. It processes papers from an Excel file and outputs a CSV file containing the paper titles and their corresponding themes.

## Prerequisites

- Python 3.x
- Excel file containing research papers with columns for Title and Abstract

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv theme_env
# On Windows:
theme_env\Scripts\activate
# On Unix or MacOS:
source theme_env/bin/activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Place your Excel file named `teacher vs ai papers.xlsx` in the same directory as the script.

2. Run the script:

```bash
python theme_extractor.py
```

3. The script will:
   - Read the Excel file
   - Preprocess the text (combining title and abstract)
   - Perform topic modeling using LDA
   - Extract dominant themes
   - Save results to `extracted_themes.csv`

## Output

The script generates a CSV file (`extracted_themes.csv`) containing:

- Paper Title
- Extracted Theme (based on dominant topic keywords)

## Notes

- The script uses LDA (Latent Dirichlet Allocation) for topic modeling
- Default number of topics is set to 12 (can be adjusted in the script)
- The coherence score is calculated and displayed during execution
- Processing time depends on the number of papers and their length
