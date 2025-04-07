import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
import pandas as pd
from theme_extractor import (
    preprocess,
    get_dominant_topic,
    get_theme_description,
    THEMES,
)
import PyPDF2
from pathlib import Path
import spacy
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
import datetime


class ThemeExtractorThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, pdf_paths, num_themes):
        super().__init__()
        self.pdf_paths = pdf_paths
        self.num_themes = num_themes
        self.is_single_file = len(pdf_paths) == 1
        self.df = None

    def start_analysis_from_df(self, df):
        """Start analysis from an existing DataFrame."""
        self.df = df
        self.start()

    def run(self):
        try:
            self.status.emit("Loading spaCy model...")
            nlp = spacy.load("en_core_web_lg")

            if self.df is None:
                # Process PDFs
                self.status.emit("Processing PDFs...")
                papers = []
                total_pdfs = len(self.pdf_paths)
                processed_count = 0

                for i, pdf_path in enumerate(self.pdf_paths):
                    try:
                        text = self.extract_text_from_pdf(pdf_path)
                        papers.append(
                            {
                                "Name of Paper": Path(pdf_path).stem,
                                "Combined_Text": text,
                            }
                        )
                        processed_count += 1
                    except Exception as e:
                        self.status.emit(f"Skipping {pdf_path}: {str(e)}")
                        continue

                    self.progress.emit(int((i + 1) / total_pdfs * 100))

                if not papers:
                    raise ValueError("No valid PDFs could be processed")

                self.status.emit(
                    f"Successfully processed {processed_count} out of {total_pdfs} PDFs"
                )
                df = pd.DataFrame(papers)
            else:
                # Use provided DataFrame
                df = self.df
                self.progress.emit(100)

            self.status.emit("Preprocessing text...")
            df["Processed_Text"] = df["Combined_Text"].apply(preprocess)

            if self.is_single_file:
                self.status.emit("Analyzing text with keyword matching...")
                df["Theme"] = df["Combined_Text"].apply(self.analyze_single_file)
            else:
                # For multiple files, use LDA
                self.status.emit("Creating dictionary and corpus...")
                texts = [text.split() for text in df["Processed_Text"]]

                if not texts or not any(texts):
                    raise ValueError("No valid text found in any of the PDFs")

                bigram = Phrases(texts, min_count=3, threshold=50)
                trigram = Phrases(bigram[texts], threshold=50)
                texts = [trigram[bigram[text]] for text in texts]

                dictionary = corpora.Dictionary(texts)
                dictionary.filter_extremes(no_below=1, no_above=0.7)
                corpus = [dictionary.doc2bow(text) for text in texts]

                self.status.emit("Building LDA model...")
                lda_model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=self.num_themes,
                    passes=100,
                    random_state=42,
                    alpha="auto",
                    eta="auto",
                    chunksize=50,
                    eval_every=None,
                    iterations=2000,
                    gamma_threshold=0.001,
                    minimum_probability=0.005,
                )

                self.status.emit("Calculating coherence score...")
                coherence_model = CoherenceModel(
                    model=lda_model,
                    texts=texts,
                    dictionary=dictionary,
                    coherence="c_v",
                    topn=10,
                )
                coherence_score = coherence_model.get_coherence()

                self.status.emit("Extracting dominant topics...")
                df["Dominant_Topic"] = df.apply(
                    lambda row: get_dominant_topic(
                        lda_model, corpus[row.name], row["Combined_Text"]
                    ),
                    axis=1,
                )

                self.status.emit("Generating theme descriptions...")
                df["Theme"] = df.apply(
                    lambda row: get_theme_description(
                        row["Combined_Text"], None, row["Dominant_Topic"]
                    ),
                    axis=1,
                )

            self.finished.emit(df)

        except Exception as e:
            self.error.emit(str(e))

    def analyze_single_file(self, text):
        """Analyze a single file using keyword matching with detailed output."""
        text_lower = text.lower()
        theme_matches = []

        # Analyze matches for each theme
        for theme_id, theme_info in THEMES.items():
            keywords = theme_info["keywords"]
            matches = [k for k in keywords if k in text_lower]
            if matches:
                theme_matches.append(
                    {
                        "theme": theme_info["name"],
                        "score": len(matches),
                        "keywords": matches,
                    }
                )

        # Sort themes by number of keyword matches
        theme_matches.sort(key=lambda x: x["score"], reverse=True)

        if theme_matches:
            best_match = theme_matches[0]
            result = f"Primary Theme: {best_match['theme']}\n"
            result += f"Confidence: {'High' if best_match['score'] >= 3 else 'Medium' if best_match['score'] == 2 else 'Low'}\n"
            result += f"Matched Keywords: {', '.join(best_match['keywords'])}\n"

            # Add secondary themes if they exist
            if len(theme_matches) > 1:
                secondary = theme_matches[1]
                result += f"\nSecondary Theme: {secondary['theme']}\n"
                result += f"Matched Keywords: {', '.join(secondary['keywords'])}"

            return result

        return "Theme: General Education\nNote: No specific theme keywords were found in the text. The content may be too general or may need different keyword matching criteria."

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file with error handling and logging."""
        log_messages = []

        def log(message):
            log_messages.append(message)
            self.status.emit(message)

        try:
            log(f"Attempting to process: {pdf_path}")

            # Try different PDF processing approaches
            approaches = [
                self._try_pypdf2_approach,
                self._try_pypdf2_alternative,
                self._try_pypdf2_simplified,
            ]

            for approach in approaches:
                try:
                    text = approach(pdf_path, log)
                    if text and text.strip():
                        log(f"Successfully extracted text using {approach.__name__}")
                        return text
                except Exception as e:
                    log(f"Approach {approach.__name__} failed: {str(e)}")
                    continue

            raise ValueError("All PDF processing approaches failed")

        except Exception as e:
            error_msg = f"Error processing {pdf_path}: {str(e)}\nLog:\n" + "\n".join(
                log_messages
            )
            self.status.emit(error_msg)
            raise ValueError(error_msg)

    def _try_pypdf2_approach(self, pdf_path, log):
        """First approach: Standard PyPDF2 processing with MediaBox error handling."""
        try:
            with open(pdf_path, "rb") as file:
                log(
                    f"Processing PDF: {os.path.basename(pdf_path)}"
                )  # Log the current file name
                reader = PyPDF2.PdfReader(file)
                text = ""
                total_pages = len(reader.pages)

                for page_num in range(total_pages):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
                    except Exception as page_error:
                        log(
                            f"Error in file '{os.path.basename(pdf_path)}' on page {page_num + 1}: {str(page_error)}"
                        )
                        continue
                return text
        except Exception as e:
            log(f"Failed to process '{os.path.basename(pdf_path)}': {str(e)}")
            return ""

    def _try_pypdf2_alternative(self, pdf_path, log):
        """Second approach: Alternative PyPDF2 processing with error recovery."""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            total_pages = len(reader.pages)

            for page_num in range(total_pages):
                try:
                    page = reader.pages[page_num]
                    # Try to get page content directly, bypassing MediaBox
                    if "/Contents" in page:
                        content = page["/Contents"]
                        if isinstance(content, list):
                            content = b"".join(content)
                        text += str(content) + " "
                    else:
                        # Try to extract text using a different method
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + " "
                        except:
                            # If text extraction fails, try to get raw content
                            if "/Resources" in page:
                                resources = page["/Resources"]
                                if "/Font" in resources:
                                    text += (
                                        "Text content available but extraction failed. "
                                    )
                except Exception as page_error:
                    log(
                        f"Warning: Alternative approach failed for page {page_num + 1}: {str(page_error)}"
                    )
                    continue

            return text

    def _try_pypdf2_simplified(self, pdf_path, log):
        """Third approach: Simplified PyPDF2 processing with minimal dependencies."""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""

            # Try to get just the first few pages
            max_pages = min(5, len(reader.pages))
            for page_num in range(max_pages):
                try:
                    page = reader.pages[page_num]
                    # Try multiple extraction methods
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
                    except:
                        # If standard extraction fails, try direct content access
                        if "/Contents" in page:
                            content = page["/Contents"]
                            if isinstance(content, list):
                                content = b"".join(content)
                            text += str(content) + " "
                except Exception as page_error:
                    log(
                        f"Warning: Simplified approach failed for page {page_num + 1}: {str(page_error)}"
                    )
                    continue

            return text


class ThemeExtractorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Research Paper Theme Extractor")
        self.setMinimumSize(800, 600)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create info section
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout()

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setPlainText(
            "Technologies Used:\n"
            "- Python 3.x\n"
            "- PyQt6 for GUI\n"
            "- spaCy (en_core_web_lg) for NLP\n"
            "- Gensim for LDA topic modeling\n"
            "- PyPDF2 for PDF processing\n\n"
            "Algorithms:\n"
            "- Latent Dirichlet Allocation (LDA)\n"
            "- N-gram detection\n"
            "- Text preprocessing with spaCy\n"
            "- Keyword-based theme matching\n\n"
            "Features:\n"
            "- PDF processing\n"
            "- Multi-document analysis\n"
            "- Customizable theme count\n"
            "- Progress tracking\n"
            "- Coherence score calculation"
        )
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Create controls section
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()

        # Theme count selector
        theme_layout = QHBoxLayout()
        self.theme_label = QLabel("Number of Themes:")
        self.theme_spinbox = QSpinBox()
        self.theme_spinbox.setRange(5, 50)
        self.theme_spinbox.setValue(15)
        theme_layout.addWidget(self.theme_label)
        theme_layout.addWidget(self.theme_spinbox)
        controls_layout.addLayout(theme_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.folder_button = QPushButton("Analyze Folder")
        self.file_button = QPushButton("Analyze Single File")
        self.folder_button.clicked.connect(self.analyze_folder)
        self.file_button.clicked.connect(self.analyze_file)
        button_layout.addWidget(self.folder_button)
        button_layout.addWidget(self.file_button)
        controls_layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        controls_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Set styles
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #040c1d;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #465a63;
                border-radius: 5px;
                margin-top: 1ex;
                background-color: #040c1d;
                color: #3561a4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #3561a4;
            }
            QPushButton {
                background-color: #3561a4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #465a63;
            }
            QPushButton:disabled {
                background-color: #465a63;
            }
            QProgressBar {
                border: 1px solid #465a63;
                border-radius: 3px;
                text-align: center;
                background-color: #040c1d;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #3561a4;
            }
            QLabel {
                color: #3561a4;
            }
            QTextEdit {
                background-color: #040c1d;
                color: white;
                border: 1px solid #465a63;
            }
            QSpinBox {
                background-color: #040c1d;
                color: white;
                border: 1px solid #465a63;
            }
        """
        )

    def analyze_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            # Check for CSV files
            csv_files = [
                f for f in os.listdir(folder_path) if f.lower().endswith(".csv")
            ]

            if len(csv_files) == 1:
                # If there's exactly one CSV file, use it
                csv_path = os.path.join(folder_path, csv_files[0])
                try:
                    df = pd.read_csv(csv_path)
                    if "Name of Paper" in df.columns and "Combined_Text" in df.columns:
                        self.status.emit(f"Using CSV file: {csv_files[0]}")
                        self.theme_spinbox.setVisible(True)
                        self.theme_label.setVisible(True)
                        self.start_analysis_from_df(df)
                        return
                    else:
                        QMessageBox.warning(
                            self,
                            "Warning",
                            "CSV file must contain 'Name of Paper' and 'Combined_Text' columns",
                        )
                except Exception as e:
                    QMessageBox.warning(
                        self, "Warning", f"Error reading CSV file: {str(e)}"
                    )

            # If no CSV or CSV processing failed, look for PDFs
            pdf_files = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        pdf_files.append(os.path.join(root, file))

            if not pdf_files:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No PDF files found in the selected folder or its subfolders.",
                )
                return

            QMessageBox.information(
                self,
                "Files Found",
                f"Found {len(pdf_files)} PDF files in the selected folder and its subfolders.",
            )

            self.theme_spinbox.setVisible(True)
            self.theme_label.setVisible(True)
            self.start_analysis(pdf_files)

    def analyze_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", "", "PDF Files (*.pdf)"
        )
        if file_path:
            self.theme_spinbox.setVisible(False)
            self.theme_label.setVisible(False)
            self.start_analysis([file_path])

    def start_analysis(self, pdf_files):
        self.folder_button.setEnabled(False)
        self.file_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.results_text.clear()

        self.pdf_files = pdf_files  # Store pdf_files as instance variable
        self.worker = ThemeExtractorThread(pdf_files, self.theme_spinbox.value())
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def analysis_finished(self, df):
        self.folder_button.setEnabled(True)
        self.file_button.setEnabled(True)

        # Display results
        results = "Analysis Results:\n\n"
        for _, row in df.iterrows():
            results += f"Paper: {row['Name of Paper']}\n"
            results += f"Theme: {row['Theme']}\n\n"

        self.results_text.setPlainText(results)

        # Save results to Excel
        try:
            output_dir = os.path.dirname(self.pdf_files[0])
            output_path = os.path.join(output_dir, "extracted_themes.xlsx")
            df[["Name of Paper", "Theme"]].to_excel(output_path, index=False)
            QMessageBox.information(self, "Success", f"Results saved to {output_path}")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not save Excel file: {str(e)}")

    def show_error(self, message):
        self.folder_button.setEnabled(True)
        self.file_button.setEnabled(True)

        # Save error log to file
        try:
            log_dir = "error_logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"error_log_{timestamp}.txt")

            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Error occurred at: {datetime.datetime.now()}\n")
                f.write("=" * 50 + "\n")
                f.write(message)

            QMessageBox.critical(
                self, "Error", f"{message}\n\nError log has been saved to: {log_file}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"{message}\n\nFailed to save error log: {str(e)}"
            )

    def start_analysis_from_df(self, df):
        """Start analysis from an existing DataFrame."""
        self.folder_button.setEnabled(False)
        self.file_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.results_text.clear()

        self.worker = ThemeExtractorThread([], self.theme_spinbox.value())
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.show_error)
        self.worker.start_analysis_from_df(df)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThemeExtractorGUI()
    window.show()
    sys.exit(app.exec())
