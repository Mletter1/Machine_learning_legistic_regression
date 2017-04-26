assignment 3 in 529
Date: April 9 2015
**********************************************
Author Matthew Letter
email mletter1@unm.edu
**********************************************
overview

I implemented a generalized function that takes
a map of {class:[vectors]} and performs the
multinomial logistic regression algorithm using
gradient descent on the map. First it takes the
map and splits it into training and testing for
10x cross validation then it proceeds to do
2000 epochs of training before moving onto
the next cross validation testing and training
set. The confusion matrix is fed into the next
run and the accuracies for each run are stored
in a list for later averaging.

math:
Given: m, the number of examples
           t, the number of unique classifications an example can have
           n, the number of attributes each example has
           η, a learning rate
           λ, a penalty term
           Δ, a t×m matrix where Δji=δ(Yi=yj) (using the delta equation as found in equation (29) in the Mitchell chapter)
           X, an m×(n+1) matrix of examples, where ∀i,Xi0=1, and Xi1 through Xin are the attributes for example i
           Y, an m×1 vector of true classifications for each example
           W, a t×(n+1) matrix of weights

           P(Y|W,X)∼exp(WXT), a t×m matrix of probability values.
           To follow the format of equations (27) and (28) in the text,
           fill the last row with all 1's, and then normalize each column to sum
           to one by dividing the each value in the column by the sum of the column.

    Then the update step for the logistic regression is

    Wt+1=Wt+η((Δ−P(Y|W,X))X−λWt)
**********************************************
how to run

print out help message:
    python main.py -h

Run logistic regression algorithm with verbose output
    python main.py -v

Run process data then logistic regression algorithm with verbose output assumes starting directory = "."
    python main.py -p

Run process data then logistic regression algorithm with verbose output assumes starting directory = arg[0]
    python main.py -p <path to dir with opihi.cs.uvic.ca>

Run logistic regression algorithm with pre-processed data assumes starting directory = "." for pickle data
and that the pickled data is in a file named "data.p"
    python main.py
**********************************************
Bugs
