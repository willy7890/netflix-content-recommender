# Netflix Content Recommendation System

A content-based recommendation system that recommends Netflix movies and TV shows based on similarities in their available metadata.

## Project Overview

This project implements a content-based filtering approach using Python and machine learning techniques. It processes Netflix titles and compares their metadata to recommend other titles with similar content categories and characteristics.

The project was developed as part of the Machine Learning Internship at Auspify Technologies.

## Objectives

- Load and clean the Netflix dataset.
- Prepare relevant content features.
- Convert text features into numerical representations using TF-IDF.
- Calculate similarities between titles using cosine similarity.
- Generate ranked recommendations for a selected movie or TV show.
- Evaluate recommendation results and test core functionality.

## Technologies Used

- **Python 3.11**
- **Pandas** — data loading and preprocessing
- **Scikit-learn** — TF-IDF vectorization and cosine similarity
- **Pytest** — automated testing
- **NumPy** — numerical operations

## How It Works

The recommendation pipeline follows these steps:

1. **Data loading:** Reads the Netflix dataset from `data/raw/Dataset.csv`.
2. **Data cleaning:** Normalizes column names, handles missing text values, and removes duplicate records.
3. **Feature preparation:** Combines available metadata, including categories, director, country, content type, and rating.
4. **Text vectorization:** Uses `TfidfVectorizer` to convert text into numerical feature vectors.
5. **Similarity calculation:** Uses cosine similarity to compare titles.
6. **Recommendation ranking:** Ranks candidate titles using text similarity and a category-overlap bonus.
7. **Output:** Displays recommended titles with metadata and similarity scores.

The system uses the metadata available in the supplied dataset. It does not currently use plot descriptions, cast information, user ratings, or individual viewing histories.

## Project Structure

```text
netflix-content-recommender/
├── data/
│   └── raw/
│       └── Dataset.csv
├── scripts/
│   └── evaluate_recommender.py
├── src/
│   └── recommender/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       └── model.py
├── tests/
│   └── test_recommender.py
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.11 or a compatible Python version
- pip
- The Netflix dataset supplied for this project

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/willy7890/netflix-content-recommender
cd netflix-content-recommender
```
### 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Recommendation System

Run the following command from the project root:

```bash
PYTHONPATH=src python -m recommender.model
```

Enter the name of a movie or TV show when prompted. The program will display recommendations based on the selected title.

Example input:

```text
Ganglands
```

The output includes recommended titles, content type, release year, rating, categories, and similarity scores.

## Evaluating the Model

Run the evaluation script:

```bash
PYTHONPATH=src python scripts/evaluate_recommender.py
```

The evaluation script checks selected titles and reports the number of recommendations, average similarity score, and category overlap.

### Evaluation Results

The following results were observed during development:

| Test title | Recommendations | Average similarity | Category overlap |
|---|---:|---:|---:|
| Ganglands | 10 | 0.7666 | 100% |
| Dick Johnson Is Dead | 10 | 0.2966 | 100% |
| Lupin | 10 | 0.9133 | 100% |

These results are from the current development run and may change if the dataset, preprocessing, or algorithm changes.

**Metric interpretation:** Cosine similarity measures similarity between feature vectors; it is not an accuracy percentage. Category overlap measures whether recommended titles share at least one category with the selected title. Category overlap alone does not establish recommendation quality, so further evaluation with a larger and more representative test set would be useful.

## Running Tests

Run the automated tests with:

```bash
PYTHONPATH=src pytest -v
```

The current test suite contains four tests covering:

- Recommendation generation
- Exclusion of the selected title from its own recommendations
- Handling of titles that are not found
- Validation of the `top_n` parameter

Latest recorded test result: **4 passed**.

## Current Limitations

- Recommendations depend on the metadata available in the dataset.
- The dataset does not include plot descriptions or cast information.
- Titles with limited or unusual metadata may receive less relevant recommendations.
- The current evaluation uses a small set of example titles and should not be interpreted as a comprehensive accuracy benchmark.
- The system does not personalize recommendations using individual user preferences or viewing history.

## Possible Future Improvements

- Evaluate more titles across different content categories.
- Compare alternative feature-weighting strategies.
- Add more relevant content metadata if a suitable dataset becomes available.
- Develop a more comprehensive evaluation procedure using manually reviewed recommendations.

## Internship Submission

**Project:** Netflix Content Recommendation System  
**Organization:** Auspify Technologies  
**Category:** Machine Learning  
**Approach:** Content-Based Filtering

Submission materials may include the source code, screenshots of successful execution, evaluation results, and the GitHub repository link.

## Author

**Wilbard Magaso**

Machine Learning Intern — Auspify Technologies
