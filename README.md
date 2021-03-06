# A Dataset of Journalists' Interactions with Their Readership: When Should Article Authors Reply to Reader Comments?

This repository contains a python script and files that list comment IDs.
It is part of our resource paper published at CIKM 2020, which is titled "A Dataset of Journalists' Interactions with Their Readership: When Should Article Authors Reply to Reader Comments?".

To support repeatability and push further research in this area, we provide a script to download a set of 38,000 of comments. The script accesses the Guardian’s Web API to download a predefined list of comments identified by their IDs. Half of these comments received a reply by a journalist, while the other half did not. Due to the provided class labels and the balanced class distribution, the comments can be easily used for supervised machine learning.
Further, we provide the comment IDs of the journalists' replies. 

The script takes comment IDs as input and retrieves the corresponding comments via the Guardian's API. 
An API key is required to access the API. You can register for a key by filling out [this short form](https://bonobo.capi.gutools.co.uk/register/developer).

In case your daily number of API calls is limited, the script stops when the limit is reached. If restarted, the script will continue from the point where it stopped.

Example usage:

```python3 GAT.py --apikey "<your_api_key_here>" --source "comment_ids_replied_to_by_the_journalist10.csv" --output "comments_replied_to_by_the_journalist.csv"```

There are three example files for testing purposes, which contain only 10 comment IDs:

```comment_ids_replied_to_by_the_journalist10.csv```, ```comment_ids_not_replied_to_by_the_journalist10.csv```, and ```comment_ids_of_journalist_replies10.csv```

Example output: 
![alt text](example_output.png "Example Output")

# Labels
We machine-labeled all comments and in addition manually labeled a subset of them. The labels are contained in the files:
```comment_ids_replied_to_and_machine-labeled_sentiment.csv```, ```comment_ids_not_replied_to_and_machine-labeled_sentiment.csv```, and ```comment_ids_replied_to_and_manual_annotations.csv```

# Word Embeddings
Our FastText Word Embeddings were trained on 60 million comments from The Guardian and can be downloaded [here](https://owncloud.hpi.de/s/8LjQz1nyFI3OZBe).

A Web browser-based visualization of the embeddings can been accessed [here](https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/julian-risch/9d6d125b7b5e49eb9b1ffacfd6de922a/raw/22b39854e2f5acb31dac1d586871d3fee7a4d0ca/1-10k-projector_config.json).

# Code
```pairwise-comment-ranking.py``` and ```pairwise-comment-ranking-data-generator.py``` contain example code that shows how to load the fasttext embeddings, how to apply them, and how to use pairs of comments as input to train a neural network model.
[FastText](https://github.com/facebookresearch/fastText/#building-fasttext-for-python) needs to be installed to use the embeddings.

# Citation
If you use our work, please cite our paper [**A Dataset of Journalists' Interactions with Their Readership: When Should Article Authors Reply to Reader Comments?**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2020dataset.pdf) published at CIKM'20 as follows:

    @inproceedings{risch2020dataset,
    title = {A Dataset of Journalists' Interactions with Their Readership: 
    When Should Article Authors Reply to Reader Comments?},
    author = {Risch, Julian and Krestel, Ralf},
    booktitle = {Proceedings of the International Conference on Information and Knowledge Management (CIKM)},
    pages = {3117--3124},
    year = {2020}
    }

# Acknowledgments
First of all, we would like to thank *The Guardian* for providing access to the comment data via their *Guardian Open Platform*.
This project was partly funded by the Robert Bosch Foundation in the context of its master class *Science Journalism* 2018.
We would like to thank the collaborating data scientist Andreas Loos from *ZEIT ONLINE* and journalist Wolfgang Richter from *Journalistenbüro Schnittstelle* for the close cooperation.
Further, we would like to thank our students Nicolas Alder, Lasse Kohlmeyer, and Niklas Köhnecke for their support throughout the project.
