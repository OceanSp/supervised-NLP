# Project Setup

## Required Data Files

Before running the notebook, you need to download the GloVe embeddings:

1. Download GloVe embeddings from Stanford NLP:

   - Visit: https://nlp.stanford.edu/data/glove.6B.zip
   - Download the zip file
   - Extract `glove.6B.100d.txt` to the project root directory

2. Alternative download methods:

   ```bash
   # Using wget
   wget https://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   ```

## Project Structure

Place the downloaded file in this structure:

- project/
  ├── notebooks/
  ├── glove.6B.100d.txt # Place the downloaded file here
  └── README.mdCopy
