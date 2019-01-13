# Disaster Response Pipeline Project
A Udacity Data Scientist Nanodegree Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

Beyond the Anaconda distribution of Python, the following packages need to be installed for nltk:
* punkt
* wordnet
* averaged_perceptron_tagger

## Project Motivation<a name="motivation"></a>

In this project I analyized thousands of real messages provided by Figure 8, sent during natural disasters either via social media or directly to disaster response organizations. I built an ETL pipeline that processes message and category data from csv files and load them into a SQLite database, which was fed to a machine learning pipeline to train, tune and save a multi-output supervised learning model. Then, my web app will extract data from this database to provide data visualizations and use the model to classify new messages for 36 categories.

Machine learning is critical to helping different organizations understand which messages are relevant to them and which messages to prioritize. During these disasters is when they have the least capacity to filter out messages that matter, and find basic methods such as using key word searches to provide trivial results. I learned the skills in ETL pipelines, natural language processing, and machine learning pipelines to create an amazing project with real world significance.





### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Important Files:
- data/process_data.py:       ETL pipeline used to process data in preparation for model building.
- models/train_classifier.py: Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle
- app/templates/*.html:       HTML templates for the web app.
- run.py:                     Start the Python server for the web app and prepare visualizations.

----------------------------------------------------------------------------------------------------------









## File Descriptions <a name="files"></a>

There are 1 notebooks available here to showcase work related to the above questions. The notebooks is exploratory in searching through the data pertaining to the questions showcased by the notebook title. Markdown cells & comments were used to assist in walking through the thought process for individual steps.

PlayerAttributeData.csv       - This file contains Player performance attributes (Overall, Potential, Aggression, Agility etc.) indexed by player id.

PlayerPersonalData.csv        - This file contains basic Player personal attributes (Nationality, Club, Photo, Age, Wage, Value etc.)

PlayerPlayingPositionData.csv - This file contains Player preferred position and ratings at all positions.

CompleteDataset.csv           - This file is the complete dataset contains all informaiton in above three dataset.


## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@haiqingzhou/fifa-18-complete-player-dataset-some-exploration-questions-2ccae897416a).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Aman Shrivastava & EA Sports for the data. You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/thec03u5/fifa-18-demo-player-dataset). Otherwise, feel free to use the code here as you would like! 
