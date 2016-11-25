# ml-kiva
Use machine learning to predict loan expiration on Kiva's site.


## Setup

1. Download the json snapshot and unzip it into this folder. https://build.kiva.org/docs/data/snapshots

2. Use import.py to generate a single csv of training data.

3. Build a model using the data. BigML, Tensorflow, etc.
ml.py, and mlChart.py are me starting to play around, but not working yet.


## Feature ideas: 

Caveat: Some features may appear to be shallow, morbid etc. but so can people's behaviour, the only way to know if they are good predictors or not is to try them.

- is the person in the photo smiling, use SimpleCSV
- is the country on a travel warning watch list
- did the country have an international disaster in the last 2 months
- does the recipient have an anglo name
- country's gdp
- country's population
