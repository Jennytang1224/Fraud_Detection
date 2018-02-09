## Premise
You are a contract data scientist/consultant hired by a new e-commerce site to try to weed out fraudsters.  The company unfortunately does not have much data science expertise... so you must properly scope and present your solution to the manager before you embark on your analysis.  Also, you will need to build a sustainable software project that you can hand off to the companies engineers by deploying your model in the cloud.  Since others will potentially use/extend your code you **NEED** to properly encapsulate your code and leave plenty of comments.

## The Data
#### NOTE: due to the sensitivity of the data, the data will not be posted github

### The "product" of fraud
Something that you will need to think about throughout this case study is how the product of your client fits into the given technical process.  A few points to note about the case of fraud:

* Failures are not created equal
    * False positives decrease customer/user trust
    * False negatives cost money
        * Not all false negative cost the same amount of money
* Accessibility
    * Other (non-technical) people may need to interact with the model/machinery
    * Manual review

The fraud problem is actually semi-supervised in a way.  You do not use the model to declare a ground truth about fraud or not fraud, but simply to flag which transactions need further manual review.  You will be building a triage model of what are the most pressing (and costly) transactions you have seen.

### Step 1: EDA
Before you start building the model, start with some EDA.

#### [Deliverable]: Look at the data
Start by looking at the data.

1. Load the data with pandas. Add a 'Fraud' column that contains True or False values depending on if the event is fraud. If `acct_type` field contains the word `fraud`, label that point Fraud.

2. Check how many fraud and not fraud events you have.

3. Look at the features. Make note of ones you think will be particularly useful to you.

4. Do any data visualization that helps you understand the data.


#### [Deliverable]: Scoping the problem
Before you get cranking on your model, think of how to approach the problem.

1. What preprocessing might you want to do? How will you build your feature matrix? What different ideas do you have?

2. What models do you want to try?

3. What metric will you use to determine success?


### Step 2: Building the Model

#### [Deliverable]: Comparing models
Start building your potential models.

**Notes for writing code:**
* As you write your code, **always be committing** (ABC) to Github!
* Write **clean and modular code**.
* Include **comments** on every method.

*Make sure to get a working model first before you try all of your fancy ideas!*

1. If you have complicated ideas, implement the simplest one first! Get a baseline built so that you can compare more complicated models to that one.

2. Experiment with using different features in your feature matrix. Use different featurization techniques like stemming, lemmatization, tf-idf, part of speech tagging, etc.

3. Experiment with different models like SVM, Logistic Regression, Decision Trees, kNN, etc. You might end up with a final model that is a combination of multiple classification models.

4. Compare their results. Pick a good metric; don't just use accuracy!


#### [Deliverable]: Model description and code
After all this experimentation, you should end up with a model you are happy with.

1. Create a file called `model.py` which builds the model based on the training data.

    * Feel free to use any library to get the job done.
    * Again, make sure your code is **clean**, **modular** and **well-commented**! The general rule of thumb: if you looked at your code in a couple months, would you be able to understand it?

2. In your pull request, describe your project findings including:
    * An overview of a chosen "optimal" modeling technique, with:
        * process flow
        * preprocessing
        * accuracy metrics selected
        * validation and testing methodology
        * parameter tuning involved in generating the model
        * further steps you might have taken if you were to continue the project.


#### [Deliverable]: Pickled model

1. Use `pickle` to serialize your trained model and store it in a file. This is going to allow you to use the model without retraining it for every prediction, which would be ridiculous.

### Step 3: Prediction script

Take a few raw examples and store them in json or csv format in a file called `test_script_examples`.


#### [Deliverable]: Prediction script

1. Write a script `predict.py` that reads in a single example from `test_script_examples`, vectorizes it, unpickles the model, predicts the label, and outputs the label probability (print to standard out is fine).

    This script will serve as a sort of conceptual and code bridge to the web app you're about to build.

    Each time you run the script, it will predict on one example, just like a web app request. You may be thinking that unpickling the model every time is quite inefficient and you'd be right; we'll remove that inefficiency in the web app.


### Step 4: Database

#### [Deliverable]: Prediction script backed by a database

You'll want to store each prediction the model makes on new examples, which means you'll need a database.

1. Set up a Postgres or MongoDB database that will store each example that the script runs on. You should create a database schema that reflects the form of the raw example data and add a column for the predicted probability of fraud.

2. Write a function in your script that takes the example data and the prediction as arguments and inserts the data into the database.

    Now, each time you run your script, one row should be added to the `predictions` table with a predicted probability of fraud.

### Step 6: Get "live" data

We've set up a service for you that will send out "live" data so that you can see that your app is really working.

To use this service, you will need to make a request to our secure server. It gives a maximum of the 10 most recent datapoints, ordered by `sequence_number`. New datapoints come in every 2-3 minutes.

```python
import requests
api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
sequence_number = 0
response = requests.post(url, json={'api_key': api_key,
                                    'sequence_number': sequence_number})
raw_data = response.json()

```

Write a function that periodically fetches new data, generates a predicted fraud probability, and saves it to your database (after verifying that the data hasn't been seen before).
