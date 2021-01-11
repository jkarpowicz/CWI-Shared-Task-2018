# Complex Word Identification (CWI) Shared Task 2018 - Jagoda Karpowicz

This Project addresses __Difiicult Words Classification Shared Tasks__ - prediction of words hard to understand by children, people with disabilities and non-native speakers for English and Spanish Language. For more information please see `ClassPojectReport.pdf`.


## Code structure

`Main.py` file contains main program execution.

`\utils\baseline.py` - python file with base features extration and baseline classification.

`\utils\improved.py` - python file with additional features extration and extended classification.

`\utils\dataset.py` - python file used reading train, dev and test data from the datasets (not attached).

`\utils\scorer.py` - used to score the predictions of the models.

`\LearningCurve.py` creates plots with baseline and improved baseline development and test datasets.

## Dataset 

DaleChallEasyWordList.txt - list of simple word used by the code.

Dataset used for training and tetsing can be found at https://sites.google.com/view/cwisharedtask2018/home


## Code execution

The program can be run by calling `Main.py`.

## Results

F1 Score for Spanish and English Dataset will be printed in the consol together with examples.
