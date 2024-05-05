# xtream AI Challenge

## Ready Player 1? 🚀

Hey there! If you're reading this, you've already aced our first screening. Awesome job! 👏👏👏

Welcome to the next level of your journey towards the [xtream](https://xtreamers.io) AI squad. Here's your cool new assignment.

Take your time – you've got **10 days** to show us your magic, starting from when you get this. No rush, work at your pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### What You Need to Do

Think of this as a real-world project. Fork this repo and treat it as if you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the data and craft your own effective solutions.

🚨 **Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* Your understanding of the data
* The clarity and completeness of your findings
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Keep This in Mind**: This isn't about building the fanciest model: we're more interested in your process and thinking.

---

### Diamonds

**Problem type**: Regression

**Dataset description**: [Diamonds Readme](./datasets/diamonds/README.md)

Meet Don Francesco, the mystery-shrouded, fabulously wealthy owner of a jewelry empire. 

He's got an impressive collection of 5000 diamonds and a temperament to match - so let's keep him smiling, shall we? 
In our dataset, you'll find all the glittery details of these gems, from size to sparkle, along with their values 
appraised by an expert. You can assume that the expert's valuations are in line with the real market value of the stones.

#### Challenge 1

Plot twist! The expert who priced these gems has now vanished. 
Francesco needs you to be the new diamond evaluator. 
He's looking for a **model that predicts a gem's worth based on its characteristics**. 
And, because Francesco's clientele is as demanding as he is, he wants the why behind every price tag. 

Create another Jupyter notebook where you develop and evaluate your model.

#### Challenge 2

Good news! Francesco is impressed with the performance of your model. 
Now, he's ready to hire a new expert and expand his diamond database. 

**Develop an automated pipeline** that trains your model with fresh data, 
keeping it as sharp as the diamonds it assesses.

#### Challenge 3

Finally, Francesco wants to bring your brilliance to his business's fingertips. 

**Build a REST API** to integrate your model into a web app, 
making it a cinch for his team to use. 
Keep it developer-friendly – after all, not everyone speaks 'data scientist'!

#### Challenge 4

Your model is doing great, and Francesco wants to make even more money.

The next step is exposing the model to other businesses, but this calls for an upgrade in the training and serving infrastructure.
Using your favorite cloud provider, either AWS, GCP, or Azure, design cloud-based training and serving pipelines.
You should not implement the solution, but you should provide a **detailed explanation** of the architecture and the services you would use, motivating your choices.

So, ready to add some sparkle to this challenge? Let's make these diamonds shine! 🌟💎✨

---

## How to run

### Challenge 1

Run the notebook [ch-1.ipynb](ch-1.ipynb). The first cell installs all the required packages. 
Necessary comments and observations are written inside the notebook.

### Challenge 2

The training and infer pipelines have been automated using docker, the pipelines have been tested with Docker 4.28.0, 
and they require the Docker compose functionality. Download docker from the official 
[website](https://docs.docker.com/get-docker/).

Follow the next steps to execute the training pipeline:
- Clone the repository. 
- Navigate to [train_pipeline](train_pipeline).
- Execute the docker-compose command:
   ```
   docker-compose up
  ```

The training pipeline will launch a docker container with a python image that installs all the required packages, binds
the volumes to access the source files, the datasets and everything needed by the application. When the image has 
finished the build procedure the python will load the dataset from the csv file, it will preprocess the data, and it 
will start training the Pytorch model.

The trained model will be saved in the ```pytorch_models``` directory with the date and the best Mean Absolute Error 
Loss obtained on the validation set. Whenever new data is inserted into the database you can launch the training again.
At the moment the fine-tuning of previously trained models is not implemented because new data would change the 
preprocessing procedure and an old model should not benefit from data normalized on a different dataset.

We can see the performance of the model in the [logs](logs) directory where a graph of the performance on the test set 
will be saved.

To use the trained model you can launch the inference pipeline. This pipeline takes a csv file with the same columns as 
the initial database and applies the preprocessing used during training. In the datasets folder there is the 
[infer_diamonds.csv](datasets%2Fdiamonds%2Finfer_diamonds.csv) file with the first 5 rows of the dataset and without 
the price column to test the inference function. The pipeline is already set up to use this file for inference, we 
suggest to modify the [infer_diamonds.csv](datasets%2Fdiamonds%2Finfer_diamonds.csv) to test the model performance on 
new data.

Follow the next steps to execute the inference pipeline:
- Navigate to [infer_pipeline](infer_pipeline).
- Execute the docker-compose command:
   ```
   docker-compose up
  ```
  
The predictions will be saved in a csv file in the [diamonds](datasets%2Fdiamonds) folder. The csv file will have the 
original dataset columns and the new column ```price``` with the predicted values. With the current configuration the 
script will search for the model with the best performance in the [pytorch_models](pytorch_models) directory.

