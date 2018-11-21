echo "Creating directories . . . "
mkdir data
mkdir plots
mkdir processed_corpus
mkdir processed_data
mkdir results

echo "Downloading data from Kaggle . . . "
kaggle datasets download -d kazanova/sentiment140
mv sentiment140.zip data/

echo "Unzipping data . . . "
cd data
unzip sentiment140.zip
cd ..

echo "Done!"