import os
from os.path import dirname

BASE_FOLDER = dirname(os.path.abspath(__file__))
EXT_LIBRARY_FOLDER = "ext"
GIBBSLDA_PATH = os.path.join(BASE_FOLDER, EXT_LIBRARY_FOLDER, "gibbslda", "lda")
BIGCLAM_PATH = os.path.join(BASE_FOLDER, EXT_LIBRARY_FOLDER, "agm-package", "bigclam", "bigclam")
DATASET_FOLDER = os.path.join(BASE_FOLDER, "datasets")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "outputs")