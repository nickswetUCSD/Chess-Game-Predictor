# CHESS GAME PREDICTOR -- READ ME
Chess games are awesome. Let's try and predict who wins! 

# Repo Anatomy
This is Nick Swetlin and Gallant Tsao's repository for our work on training models to predict the outcome of chess games! 
We train two types of models: Soft-SVMs and HistGradBoostClassifiers, courtesy of Sci-Kit Learn -- read our paper ***report.pdf*** for more info.

- ***data***: Folder that contains a .csv file of 6.27 million online games from lichess.org.
- ***preliminary_work***: Folder that contains an .ipynb file of our initial data investigation and exploration, as well as visualizations of our methods.
- ***main.py***: Script that runs our experiment and returns results.
- ***utils.py***: Library of functions that contains all necessary dependencies and imports for ***main.py***

# How To Use
To start, download the repo, change directories to the main repo, and run ***main.py***
```
python main.py
```
---

Entering the above performs a quick version of our experiment (using only 10k games as training data).
If you would like to recreate our full experiment on 100k rows, please change the following section of main.py:

```
if __name__ == '__main__':
    main(10000)
```

to, instead say:

```
if __name__ == '__main__':
    main(100000)
```

---

We would like to thank our professor Jingbo Shang for enabling us with the techniques and knowledge to pursue this project.
