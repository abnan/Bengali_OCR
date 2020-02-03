# Bengali_OCR
My transfer learning approach for Bengali OCR for Kaggle Competetion: https://www.kaggle.com/c/bengaliai-cv19

![Example](https://github.com/abnan/Bengali_OCR/blob/master/images/example.png "Example")

Bengali has 49 letters (to be more specific 11 vowels and 38 consonants) in its alphabet, there are also 18 potential diacritics, or accents. This means that there are many more graphemes, or the smallest units in a written language. The added complexity results in ~13,000 different grapheme variations (compared to English’s 250 graphemic units).

# Files
1. [EDA.ipynb](https://github.com/abnan/Bengali_OCR/blob/master/EDA.ipynb) - Some EDA and initial training
2. [TrainBook.ipynb](https://github.com/abnan/Bengali_OCR/blob/master/TrainBook.ipynb) - Actual training and results on train data
3. [preprocess.py](https://github.com/abnan/Bengali_OCR/blob/master/preprocess.py) - Save images extracted from .parquet files


# Confusion matrices
Despite the unbalanced class distribution, the results came out pretty well!

Vowels:
![Vowels](https://github.com/abnan/Bengali_OCR/blob/master/images/vowel.png "Vowels")

Consonants:
![Consonants](https://github.com/abnan/Bengali_OCR/blob/master/images/consonant.png "Consonants")

Base grapheme:
Since there are ~110 combinations, only showing the diagonal of the matrix is feasible
![Grapheme](https://github.com/abnan/Bengali_OCR/blob/master/images/grapheme.png "Grapheme")

# Todo:
* Clean-up files
* Add test data results
* Try weighted loss or under/over-sampling for under/over-represented classes
