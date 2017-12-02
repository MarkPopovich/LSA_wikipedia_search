# Latent Semantic Analysis
## Wikipedia Search Enginge

### By Mark Popovich

## Introduction

Latent Semantic Analysis refers to the combination of Term Frequency-Inverse Document Frequency and Truncated Singular Value Decomposition. Together, these two techniques make possible powerful and expedient analysis of complexities and principal components in text documents. The implementation of cosine similiarity allows for the measurement of the distance between two documents principal components, allowing search functionality over a corpus of text.

This project utilizes these techniques to understand the principal components of two major categories of wikipedia ("Machine Learning" and "Business Software" in this case, although the code can be adjusted for any two categories). By providing either a search term, such as "Artificial Intelligence" or "Microsoft Word", a list of results is returned. In this implementation, the top five titles are printed by the script. A model was additionally trained using Logistic Regression on the full text corpus. This model can be used to predict which major category a body of text is derived from. 

Finally, a MongoDB server was initialized to hold the corpus of text after it was pulled from Wikipedia's API. The full data contains the primary category, a list of all subcategories of a particular text, the articles title, the articles internal wikipedia ID, and the full text pulled from the API. 

## Notebooks

#### wikipedia_category_search.ipynb

This notebook runs the functions to pull every page from a specific category, including all of the pages found in it's subcategories. It ensures that only unique pages are returned and is optimized to work over any category on wikipedia, including those that contain circular category paths (such as was the case for "Business Software"). Code is also contained to upload the data to the MongoDB and store it. 

NOTE: Due to limits on the Wikipedia API and the number of individual calls this code is making, it may take several minutes to run, depending on category size. 

#### semantic_search_wiki.ipynb

This notebook performs the text vectorization using TF_IDF and Truncated SVD to produce representative matrices of the text documents. 

These raw matrices are then implemented to create title search functionality, and several examples of successful search results are returned. A handful of the search results are compared to Google's search engine when searching for the same term + "wikipedia". The results are surprisingly successful and parallel googles own search engine. 

#### predicting_wikipedia_category.ipynb

A Logistic Regression model is trained on the principal component analysis done in the previous notebook. Text documents are broken into train and test groups, and the model performs surprisingly well. 

At .97 R Squared on both train and test, there is some worry that data leakage could be occuring in this implementation. For example, the text pulled from the wikipedia API commonly includes the text page's primary category at its end, written as "Category: x". This could be affecting the model by making the problem much simpler than it otherwise would be. 

Future revisions to this project will investigate this problem and see what effect removing these category tags would have. 