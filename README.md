# Text-Classification-Naive-Bayes
Naive Bayes-based models to do text classification on IMDB dataset

I experiment with three different ways to represent the documents. “Representation”
means how you convert the raw text of a document to a feature vector. 
I use a sparse representation of the feature vectors which is based on a dictionary
that maps from the feature name to the feature value. For each representation, there is
exactly 1 feature per unique token in the vocabulary, but the value of that feature change.

## Document Representations
1. Binary Bag-of-Words: Each document should be represented with binary features,
one for each token in the vocabulary.

2. Count Bag-of-Words Instead of having a binary feature for each token, I
keep count of how many times the token appears in the document, a quantity known
as term frequency and denoted tf(d, v).

3. TF-IDF Model The final representation use the TF-IDF score of each token.
The TF-IDF score combines how frequently the word appears in the document as well
as how infrequently that word appears in the document collection as a whole. 

##  Naive Bayes Experiment
I build three Naive Bayes classifiers, each one using one of
the above document representations, and compare their performances on the test dataset.

The prediction rule for Naive Bayes is: \
![image](https://user-images.githubusercontent.com/53811688/227085100-d90c2957-afb1-4817-814b-c4a136cece39.png)

Equations which I implement to build my Naive Bayes classifier: \
![image](https://user-images.githubusercontent.com/53811688/227085161-d865caad-bda9-4329-bd5a-92113e5bf345.png)
 \
![image](https://user-images.githubusercontent.com/53811688/227085303-e63dc95a-dc8c-478d-b4ea-d5f17b487641.png)
\
I implement Laplace smoothing on P(v | y) as follows:\
![image](https://user-images.githubusercontent.com/53811688/227085233-1e890956-cc77-465c-8f84-f2af76de6410.png)
where k is a hyperparameter which controls the strength of the smoothing.

The following equation is implemented to make prediction: \
![image](https://user-images.githubusercontent.com/53811688/227085790-43068b04-3c73-4b4d-8c95-9c22c6eed93f.png)

##  Experiment Results

![image](https://user-images.githubusercontent.com/53811688/227086154-a90aacb6-1b9d-49f8-85f9-1b9ec0600315.png) 

We can see that the most complicated document representation did not get the best results.
k helps us in a way that posterior probability does not suddenly drops to zero
when there is an additional word not in vocabulary and has p(y|v) = 0. It does
this by giving this word a small non-zero probability for both classes.
As the value of k goes to infinity, the p(v|y) will go constant
value which is 1/|V |. \
So, our conditional probability p(v|y) (likelihood) will be very similar. Then,
our posterior probabilities p(y|v) will also be similar since posterior = likelihood\*prior.
I assume priories are very close. Then, our validation accuracies will
be around 0.5 for balanced data set for the binary class problem.




