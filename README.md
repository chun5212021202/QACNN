# ACM-Net
Attention CNN Matching Net
6 folders and 1 main script should be included :

  * preprocess
    * plot2vec.py
    * qa2vec.py
    
  * word_vec
    * register.json
    * glove.42B.300d.json
    
  * raw_data
    * plot
      * IMDB_KEY.split.wiki
    * question
      * qa.json
    
  * output_data
    * plot
    * question
    
  * utility 
    * utility.py    
    
  * model
    * MODEL.py
    
  * main.py

usage
-------
1. put the data given by MovieQA into raw_data folder (including IMDB_KEY.split.wiki and qa.json)
2. download GloVe word2vec and put it into word_vec folder
   (GloVe: https://drive.google.com/file/d/0B3UsyrHYzsTzekMtbndMZXB1Ync/view?usp=sharing)
3. run 2 scripts (plot2vec.py & qa2vec.py), and all the preprocessed data will automatically be saved to output_data folder
4. run main.py
