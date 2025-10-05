# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository processes SurveyMonkey survey data for the PORTRAIT project, scoring multiple psychological assessments (PHQ-9, BAI, OCI-R, STAI, BFI, ASSIST) and generating classifications for mental health research.

## Key Commands

### Testing
```bash
pytest tests/test_scoring.py
```

### Running the Scoring Pipeline
```bash
python scoring.py -o output_file.csv
```

The script reads from `data/PORTRAIT_v3_updated.csv` by default and outputs processed scores.

### Working with the Notebook
The main workflow is in `SurveyMonkeyPreprocess.ipynb`. Open with Jupyter:
```bash
jupyter notebook SurveyMonkeyPreprocess.ipynb
```

## Architecture

### Core Components

1. **scoring.py**: Refactored scoring module with a single `score_survey()` function
   - Processes raw CSV data with Spanish headers
   - Calculates scores for 6 psychological instruments
   - Returns DataFrame with totals, classifications, and risk assessments
   - Can be imported or run as CLI

2. **SurveyMonkeyPreprocess.ipynb**: Main analysis notebook
   - Data cleaning (duplicate removal, user code standardization)
   - Validation of all instrument responses
   - Score computation and classification
   - Statistical analysis and visualizations
   - Exports to Excel and CSV formats

3. **tests/test_scoring.py**: Pytest suite validating scoring accuracy against known results for 7 test users

### Data Flow

1. Raw CSV → `data/PORTRAIT_last.csv` (input)
2. Cleaning → `data/PORTRAIT_last_updated.csv` (intermediate)
3. Scoring → `results_surveyMonkey_Processed.{csv,xlsx}` (output)

### Psychological Instruments

The system scores 6 standardized instruments with specific column ranges and scoring rules:

- **PHQ-9** (columns 89-97): Depression severity (range 0-3)
- **BAI** (columns 119-139): Beck Anxiety Inventory (range 0-4, requires -1 adjustment)
- **OCI-R** (columns 140-157): Obsessive-Compulsive Inventory (range 0-4)
- **STAI** (columns 99-118): State-Trait Anxiety (range 0-3, has reverse-scored items [0,5,6,9,12,15,18])
- **BFI** (columns 158-201): Big Five personality traits (range 1-5, subscale-based with reverse scoring)
- **ASSIST** (columns vary): Substance use risk assessment (complex multi-question structure)

Each instrument has:
- Specific header text in Spanish used to locate columns
- Validation rules for response ranges
- Custom scoring algorithms (sum, reverse scoring, subscales)
- Classification thresholds (e.g., PHQ: <5 minimal, 5-9 mild, 10-14 moderate, etc.)

### Important Implementation Details

**Column Finding**: Uses `find_column_index()` to locate instrument boundaries by matching exact Spanish text in row 1 (subheader) or row 2 (subsubheader).

**BFI Subscales**: 5 personality traits computed as averages of specific items with reverse scoring applied per subscale. Binary classification uses median cutoffs (e.g., Extraversion: 3.42).

**ASSIST Scoring**: Most complex instrument - sums responses from questions 2-7 for each substance (questions spaced 9 columns apart), excludes question 5 for tobacco. Risk levels have substance-specific thresholds.

**Gender-based STAI Classification**: Uses different thresholds for males (≥37) and females (≥29) to classify as "High" anxiety.

**Data Structure**: First 3 rows are headers (sequence numbers, subheader, subsubheader). User data starts at row index 3.

### User Exclusions

- Hardcoded list of users to remove: `USERS_TO_REMOVE` in notebook
- Duplicate handling: `DUPLICATES_TO_REMOVE` maps user codes to respondent IDs
- Special case: "02E1T" → "O2E1T" (character correction)

## Output Notes

- Results include binary BFI classifications (Low/High based on median cutoffs)
- ASSIST outputs both raw scores and risk categories (No requiere intervención, Recibir intervención breve, Tratamiento más intensivo)
- On Windows, the pipeline copies results to `W:\Portrait\SVM\data\`, on macOS to `/Volumes/mgialou/Portrait/SVM/data/`