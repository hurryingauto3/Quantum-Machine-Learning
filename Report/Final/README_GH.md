![image](images/title2.jpg)

Abstract
========

This paper aims to explore the nuances of Quantum Machine Learning by
discussing the many intricacies involved in the field. The paper begins
by discussing the techniques present in modern day machine learning
practices. It goes over surface level descriptions of Supervised and
Unsupervised learning. The paper further explores two classical machine
learning algorithms - Gradient Descent and Support Vector Machines. The
paper explores the mathematics of support vector machines to build a
basis for the quantum counterpart of the algorithm. The algorithms
perform differently and solve a different class of problems entirely
thus maintaining a diverse discussion within the context of this paper.

The paper then moves on towards exploring the quantum counterparts of
the algorithms explored and discussed before the section. Here the many
techniques of quantum machine learning relevant to these algorithms are
discussed while, however, some technicalities are omitted due to limited
scope of this paper. The paper attempts at informing the generalities
present within quantum machine while also acknowledging the broad and
open ended scope of the field in present day. The paper finally creates
a general description based on the two quantum machine learning
algorithms discussed.

Introduction
============

The data that we observe can be generally explained by an underlying
process. However, the vast amount of data that we observe around us
leaves us benign of any patterns that may emerge from that data. For
instance, we can observe that consumer behaviors are cyclical and their
purchase histories follow a certain trend. To expound, we can see that
people do no buy warm clothes all year round, or consume the same food
all year round. While these trends are easy to observe, humans tend to
miss many trends that machines can easily detect.

Machines are extremely good at crunching data, thus using data to
approximate a certain trend within it makes obseving those trends much
easier. While approximations may not preesent us the complete picture
due to the many irregularities present within the data, we are still
able to infer a certain trend from it. This is where machine learning
comes in. These trends or patterns may help us understand the
correlations within the data and by consequence help us infer a
causation for that trend. This allows us to then predict human behavior,
and other trends in data based on the assumption that these trends are
consistent and are not subject to a lot of change [@BOOK:3].

What is Machine Learning?
-------------------------

*Machine Learning (ML)* is the field of study that involves techniques
that allow machines to learn from large samples of data. Machine
Learning is a broad term that encompasses a wide variety of techniques
to characterize and classify data that then aids us in making decisions
by predicting outcomes of new data. Generally Machine Learning involves
training a machine using a learning algorithm that takes data set as
input outputs a certain classification, ir characterisitic of the data.
Machine learning solutions are currently widespread in the classical
paradigm of computing, and generally rely on classical data
sets.[@inproceedings]

### Learning Methodologies

In terms of learning mathodologies, machine learning is generally
divided into three categories:

-   **Supervised Learning**

    This method involves the use of a data set that is comprised of
    input and corresponding output vectors. A very common dataset of
    this category is used in training machines to recognize handwritten
    digits. Once the machine goes through enoug data, it becomes aware
    of trends within the handwritten data and is then capable of
    recognizing new handwritten data and map it to the right digit.
    [@BOOK:3]

-   **Unupervised Learning**

    This method involves the use of unstructured training data that
    consists of a set of input vectors $x$ without any corresponding
    target values. The goal may be to discover groups of similar
    examples within the data, where it is called *clustering*, or to
    determine the distribution of data within the input space, known as
    *density estimation*, or to project the data from a high-dimensional
    space down to two or three dimensions for the purpose of
    visualization.[@BOOK:3]

-   **Reinforcement Learning**

    This method involes the use of a reward system that trains the
    machine to better optimize itself for better predictions. The
    machine is given input vectors that it must utilize to discover
    patterns and trends within the data via trial and error. There is a
    sequence of states and actions in which the learning algorithm is
    interacting with its environment. In many cases, the current action
    not only affects the immediate reward but also has an impact on the
    reward at all subsequent time steps. For instance, by using
    appropriate reinforcement learning techniques a *neural network* can
    learn to play the game of backgammon to a high standard. Here the
    network must learn to take a board position as input, along with the
    result of a dice throw, and produce a strong move as the output.
    This is done by having the network play against a copy of itself for
    a large number of games.[@BOOK:3]

Classical Machine Learning Algorithms
=====================================

Gradient Descent
----------------

Machine Learning often relies on optimization to obtain results that are
to make use of complex classification problems, such as curve fitting,
pattern recognition, etc. These are built upon a cost/loss function that
makes use of the present data. The purpose of a cost function is to
optimize the present function so that it presents us with the most
accurate classification of the data. *Gradient Descent*.

Gradient Descent uses approximation and calculus to find the global
extremes of a function, <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode" align=middle width=31.99783454999999pt height=24.65753399999998pt/>, where <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> is an N-Dimensional vector.
The purpose of using gradient descent instead of differentiation is
simple. Not all curves are well defined functions and thus through a
simple recursive algorithm, finding the extreme values of the curve
become relatively easy.

Let us begin by making use of analogy. Suppose you're at the top of the
mountain and want to get to the bottom in the least amount of steps.
Since you're in a 3D space, you can only move in the x, y or some
combination of the two vectors. Through, Gradient Descent you can find
the best step which has the highest rate of decrease in altitude. This
process is then repeated until the decrease caused by the steepest step
is approaches a very small value <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/9389bd2cbd8c8cf96370f66d718b4f72.svg?invert_in_darkmode" align=middle width=25.570741349999988pt height=21.18721440000001pt/> Random citation [@BOOK:1]

The mathematical definition of this algorithm is as follows:
<p align="center"><img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/2af08aa481c8d6456a8d44129314150f.svg?invert_in_darkmode" align=middle width=302.12365919999996pt height=16.438356pt/></p>

Support Vector Machines
-----------------------

Machine Learning algorithms are also good at classification problems.
This involves looking at a data set and identifying *clusters* or
similar characterisitics within the data to group and classify it.

Support Vector Machines (SVM) is a supervised machine learning algorithm
used for linear discrimination problems. Our goal is to find an optimal
*hyperplane* which maximizes the margin such that it discriminates
between classes of feature vectors and is used as a decision boundary
for future data classification. The SVM is formulated as maximizingthe
distance between the hyperplane and closest data points called support
vectors. The objective function could be convex or non-convex depending
on the kernel used in SVM algorithm.[@BOOK:1]

![image](images/hyperplane.png)

We have l training examples where each example x are of D dimension and
each have labels of either y=+1 or y= -1 class, and our examples are
linearly separable. Then, our training data is of the form,

<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/e187a9ac17643462c5faecf4772fc9ba.svg?invert_in_darkmode" align=middle width=52.144314749999985pt height=24.65753399999998pt/> where i = 1 \... L, <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/a7c8560a49c77da4aa7f3ef9046f2599.svg?invert_in_darkmode" align=middle width=86.59151984999998pt height=24.65753399999998pt/>, x <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode" align=middle width=10.95894029999999pt height=17.723762100000005pt/> <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/f224ffe03d43033b6cf0879cfa55bacc.svg?invert_in_darkmode" align=middle width=23.71058414999999pt height=27.6567522pt/>

If the number of input features is 2, then the hyperplane is just a
line. If the number of input features is 3, then the hyperplane becomes
a two-dimensional plane. It becomes difficult to visualize when the
number of features exceeds 3. Support vectors are data points that are
closer to the hyperplane and influence the position and orientation of
the hyperplane. Using these support vectors, we maximize the margin of
the classifier. Deleting the support vectors will change the position of
the hyperplane.

We define a linear discriminate function as;

y = f(x) = w. x + b

where w is the p-dimensional weight vector which is perpendicular to the
hyperplane and b is a scalar which is the bias term. Adding the offset
parameter b allows us to increase the margin. If b is absent, then the
hyperplane is forced to pass through the origin, restricting the
solution. The hyperplanes in the image can be described by equation:

![image](images/svm.png)

<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/8c25ac445d4aa3b57909fda5fb04081f.svg?invert_in_darkmode" align=middle width=101.71311644999999pt height=22.831056599999986pt/> for <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710331149999991pt height=14.15524440000002pt/>=+1\
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/31380457e8cb648758a42ca98e83bbcd.svg?invert_in_darkmode" align=middle width=101.71311644999999pt height=22.831056599999986pt/> for <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710331149999991pt height=14.15524440000002pt/>=-1

We combine above two equations and we get;

<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/0985e86407afaf7e58584bf9afa4af64.svg?invert_in_darkmode" align=middle width=143.55574364999998pt height=24.65753399999998pt/> for <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width=12.710331149999991pt height=14.15524440000002pt/>=+1,-1\

The two hyperplane H1 and H2 passing through the support vectors of +1
and -1 class respectively, so:

w.x+b=-1 :H1\
w.x+b=1 :H2

The distance between H1 hyperplane and origin is <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/6ba354783cf95b365b94abaa2686f236.svg?invert_in_darkmode" align=middle width=43.15545795pt height=33.20539859999999pt/>
and distance between H2 hyperplane and origin is <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/bbb519a10a3d8ef935fc96c50581fce1.svg?invert_in_darkmode" align=middle width=32.8814376pt height=33.20539859999999pt/>.
So, margin can be given as

M=<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/bbb519a10a3d8ef935fc96c50581fce1.svg?invert_in_darkmode" align=middle width=32.8814376pt height=33.20539859999999pt/> - <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/6ba354783cf95b365b94abaa2686f236.svg?invert_in_darkmode" align=middle width=43.15545795pt height=33.20539859999999pt/>\
M=<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/81d19b4237790b487b3a64b5c59db16a.svg?invert_in_darkmode" align=middle width=17.62756545pt height=27.77565449999998pt/>

Where M is nothing but twice of the margin. So margin can be written as
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/7112ae9e580ad2e9012609fba1537af2.svg?invert_in_darkmode" align=middle width=17.62756545pt height=27.77565449999998pt/>. As, optimal hyperplane maximize the margin, then the
SVM objective is boiled down to fact of maximizing the term
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/7112ae9e580ad2e9012609fba1537af2.svg?invert_in_darkmode" align=middle width=17.62756545pt height=27.77565449999998pt/>, Maximizing this term is equivalent to saying we are
minimizing <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/cd4df3b6227740672a7aba8b321866b2.svg?invert_in_darkmode" align=middle width=21.34329449999999pt height=24.65753399999998pt/> i.e. ( min(<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/cd4df3b6227740672a7aba8b321866b2.svg?invert_in_darkmode" align=middle width=21.34329449999999pt height=24.65753399999998pt/>)) or we can say
min(<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/61f84fdcc23b14b6e62816de57449a8d.svg?invert_in_darkmode" align=middle width=31.8513459pt height=36.460254599999985pt/>) such that <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/0985e86407afaf7e58584bf9afa4af64.svg?invert_in_darkmode" align=middle width=143.55574364999998pt height=24.65753399999998pt/> for i
=1\...l

SVM optimization problem is a case of constrained optimization problem,
and it is always preferred to use dual optimization algorithm to solve
such constrained optimization problem. That's why we don't use gradient
descent.

Lagrange method is required to convert constrained optimization problem
into an unconstrained optimization problem. The goal of the above
equation is to get the optimal value for w and b. So using Lagrange
multipliers we can write the above expression as;

L = <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/1206e83a7abb2c23bedd96a47e78232a.svg?invert_in_darkmode" align=middle width=244.6951287pt height=36.460254599999985pt/>\

<p align="center"><img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/f8941b3b5949b54872076548cf96b138.svg?invert_in_darkmode" align=middle width=302.55218729999996pt height=47.93392394999999pt/></p>
Now, we take the partial derivative of it with respect to w, b and
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.58908224999999pt height=22.831056599999986pt/>.

<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/d9b052a5a056a47096d803a2723d57f3.svg?invert_in_darkmode" align=middle width=17.547073499999996pt height=28.92634470000001pt/> =
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/d14ba30fbc6ad6a9fc8142bd88707ade.svg?invert_in_darkmode" align=middle width=148.10878664999998pt height=32.51169900000002pt/>\
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/73edb52abb453816af2c158173b912f9.svg?invert_in_darkmode" align=middle width=15.524604149999996pt height=28.92634470000001pt/> =
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/9cc6062f570f4b62c8e2cf990c916920.svg?invert_in_darkmode" align=middle width=193.06951949999998pt height=32.51169900000002pt/>\
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/1a6713eb39a22dee7c1e896172915a79.svg?invert_in_darkmode" align=middle width=13.508587949999995pt height=28.92634470000001pt/> = <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/260b5b771cd6a97b10d541e49b7894b1.svg?invert_in_darkmode" align=middle width=112.81094714999998pt height=32.51169900000002pt/>

From above we get:

w = <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/c5bc6ef2a65bc15d5e58175c9b22139e.svg?invert_in_darkmode" align=middle width=84.84801269999998pt height=32.51169900000002pt/>\
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/260b5b771cd6a97b10d541e49b7894b1.svg?invert_in_darkmode" align=middle width=112.81094714999998pt height=32.51169900000002pt/>

From the above formulation we are able to find the optimal values of w
only and it is dependent on <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.58908224999999pt height=22.831056599999986pt/>, so we need to also find the
optimal value of <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.58908224999999pt height=22.831056599999986pt/>. Finding the optimal value of b needs both w
and <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.58908224999999pt height=22.831056599999986pt/>. Hence, finding the value of <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.58908224999999pt height=22.831056599999986pt/> is important for
us. Therefore, we do some algebraic manipulation:

We substitute the value of w into equation 1.

<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/e2907efb9a7cda177a2b2c7054d50fe8.svg?invert_in_darkmode" align=middle width=590.59909095pt height=42.12248040000001pt/>

<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/c930bbff186713a73e826098c5d1b889.svg?invert_in_darkmode" align=middle width=18.030320549999992pt height=22.465723500000017pt/>
=<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/e8c72b0870f7a54065bc93e8877bbdc7.svg?invert_in_darkmode" align=middle width=605.8670705999999pt height=42.12248040000001pt/>

Since our constraint was <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/eab014a92457c6170e1e49c59fffc541.svg?invert_in_darkmode" align=middle width=46.65232604999999pt height=22.831056599999986pt/> and
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/91b40eeba7b72fabaab4036c86452516.svg?invert_in_darkmode" align=middle width=116.54906834999998pt height=32.51169900000002pt/>=0 so that term becomes zero. The
first two terms get subtracted and after simplifying we have:

<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/c930bbff186713a73e826098c5d1b889.svg?invert_in_darkmode" align=middle width=18.030320549999992pt height=22.465723500000017pt/>
=<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/828d9de9d8ee2fcf09f8ab96a3fabd93.svg?invert_in_darkmode" align=middle width=266.93083394999996pt height=32.51169900000002pt/>

The optimization depends on the dot product of pairs of samples i.e.
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/c705cfb9dea51eec517fab12d0945dc4.svg?invert_in_darkmode" align=middle width=42.239264099999986pt height=14.611911599999981pt/>

Now if the samples of our classes are not linearly separable then we
transform our vectors to some other space and maximize the dot product
of the transformation <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/f3ce60618988671298c0bc22f252ce93.svg?invert_in_darkmode" align=middle width=88.22111099999998pt height=24.65753399999998pt/>. Alternatively we use a
function <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/c331c0c86eab8e445d60bf0a8ad0ab74.svg?invert_in_darkmode" align=middle width=176.55623534999998pt height=24.65753399999998pt/> such that we don't need to
know the transformation and this function gives us the dot product of
the transformations in some space. This function in the context of SVMs
is called a Kernel Function.

<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/c930bbff186713a73e826098c5d1b889.svg?invert_in_darkmode" align=middle width=18.030320549999992pt height=22.465723500000017pt/>
=<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/1e2f8961e2f6ccd09795e9ecadec67f4.svg?invert_in_darkmode" align=middle width=302.9810486999999pt height=32.51169900000002pt/>

There are many kernel functions in SVM, so how to select a good kernel
function is also a research issue. However, for general purposes, there
are some popular kernel functions: 

1.  Linear kernel: <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/70e6ac10278e7356c01402626e1c818f.svg?invert_in_darkmode" align=middle width=135.45718229999997pt height=27.6567522pt/>

2.  Polynomial kernel:
    $K (x_i , x_j) = (\gamma x_i^T x_j + r)^d , \gamma > 0$

3.  RBF Kernel:
    $K (x_i , x_j) = exp(-\gamma ||x_i - x_j||^2) , \gamma > 0$

4.  Sigmoid kernel:<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/59338e65c2ea5b64c5f1c299779fb20c.svg?invert_in_darkmode" align=middle width=208.54377885pt height=27.6567522pt/>

\*Here, <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode" align=middle width=9.423880949999988pt height=14.15524440000002pt/>, r and d are kernel parameters.[@BOOK:3]

Quantum Machine Learning Algorithms
===================================

Quantum Gradient Descent
------------------------

As seen in class, the Quantum Computing model was built up from the
Classical model through a series of additions. In this section we will
attempt to build up on the Classical model of the Gradient Descent
algorithm by first introducing the *Stochastic Gradient Descent*, then
moving on to discussing the various interpretations of a Quantum
Gradient Descent that we observed during our research.

### Stochastic Gradient Descent

As we do in complexity theory it is important to consider the
performance of our algorithm on a large value of N. This tells us how
easy it is to scale the algorithm for bigger purposes. As data sets get
larger, it becomes increasingly time consuming to perform Gradient
Descent Algorithms on it as the algorithm iterates over the entire data
set. Stochastic Gradient Descent makes use of random probabilistic
approach of finding the minima of the function. Therefore, instead of
performing the calculations over each point, a few samples are selected
and the training model learns much quicker over larger data sets as
argued by Wilson in [@BOOK:4][@BOOK:5]

Quantum Support Vector Machines
-------------------------------

During Research we encountered many versions of Quantum Support Vector
Machines (QSVM) where each academic publication had its own approach
with its own corresponding advantages and disadvantages. Implementing
SVMs using quantum algorithms can result in exponential speedup over the
classical implementations, potentially bringing the originally
polynomial complexity down to a logarithmic complexity.

### Support Vector Machines via Grover's Algorithm

**Grover's Algorithm**

Grover's Algorithm is a searching algorithm that retrieves an element
from an unordered quicker than it is possible using classical
techniques. It offers a quadratic speed up on conventional search
algorithms. Let us consider a blackbox function <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode" align=middle width=31.99783454999999pt height=24.65753399999998pt/> where <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/a3936e3017c9b788004df0b3d8fb4e43.svg?invert_in_darkmode" align=middle width=62.134673699999986pt height=24.65753399999998pt/>
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/35cb88e174fb5199c8bb8dd4ab842676.svg?invert_in_darkmode" align=middle width=18.52743584999999pt height=22.831056599999986pt/> such that <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/97631ece6f9df4f6c748c0189444dfc2.svg?invert_in_darkmode" align=middle width=46.812115349999985pt height=22.831056599999986pt/> and <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/2d2d4cc98531d7b456a3201583a538dc.svg?invert_in_darkmode" align=middle width=69.06107834999999pt height=24.65753399999998pt/>. In order, for it to
be invertible we must establish a function creates a one to one function
for each bit string in the <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/6bd87d9e2f456bcede6b5418622a42a6.svg?invert_in_darkmode" align=middle width=19.86537134999999pt height=27.6567522pt/> *(*hilbert space).

Suppose we have an <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/9bd9f68c2da2e9fcbc71324419137c73.svg?invert_in_darkmode" align=middle width=33.654089699999986pt height=22.831056599999986pt/> string where <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/ef0c93ae4b4693b5e5c3d723b739432b.svg?invert_in_darkmode" align=middle width=81.19666169999999pt height=24.65753399999998pt/>. We apply a
Hadamard Transform on <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/cf097b93aa55d2a1faf4b6af48dd122d.svg?invert_in_darkmode" align=middle width=37.578193949999985pt height=26.17730939999998pt/> to obtain an equal
superimposed state:
<p align="center"><img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/fe9f5fcea010c637bd996585ea82af1f.svg?invert_in_darkmode" align=middle width=122.47071044999998pt height=47.159971649999996pt/></p> To this
we apply *Grover's Diffusion Operator* <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/8abdece9776919e07ba13ef5119c4162.svg?invert_in_darkmode" align=middle width=54.47948054999999pt height=29.150579699999998pt/> times, which
essentially consists of applying an oracle O, Hadamard Transformations
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/f2c4edf657812c20ebe2403c5f059568.svg?invert_in_darkmode" align=middle width=39.89163089999999pt height=29.1911268pt/>, and a conditional phase shift on the states with an
exception of <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/6c7b0779fae1ce1ee269ac75a7ee4e9d.svg?invert_in_darkmode" align=middle width=19.178149649999988pt height=24.65753399999998pt/>
<p align="center"><img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/5d7fea66933d8c749e9067cfd119b7b3.svg?invert_in_darkmode" align=middle width=138.12138945pt height=19.526994300000002pt/></p> Applying
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/f2c4edf657812c20ebe2403c5f059568.svg?invert_in_darkmode" align=middle width=39.89163089999999pt height=29.1911268pt/> again
<p align="center"><img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/c28a0cc0c5490a993658602933805aad.svg?invert_in_darkmode" align=middle width=272.6522667pt height=19.5270702pt/></p>
Where <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/7a4b107ddfaca0d7066a8787ddacc320.svg?invert_in_darkmode" align=middle width=22.256529899999986pt height=24.65753399999998pt/> is the super positioned state.[@BOOK:6]

The simplest Quantum SVM makes use of an altered Grover's search
algorithm that performs an exhaustive search in the cost space. This
search is based on the minimum searching algorithm stated in
[@Article3]. The Searching Algorithm, aims to find the index of an O(N)
array where T\[n\] is minimum. Similar, to Grover, the array is probed
<img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/8abdece9776919e07ba13ef5119c4162.svg?invert_in_darkmode" align=middle width=54.47948054999999pt height=29.150579699999998pt/> times, which provides a quadratic speed up with a
probability of <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/47d54de4e337a06266c0e1d22c9b417b.svg?invert_in_darkmode" align=middle width=6.552545999999997pt height=27.77565449999998pt/>. The algorithm can be directly cited from
[@BOOK:6] as follows:

1.  Choose threshold index between <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/cab109d9894279e9dc1a3afeaff59245.svg?invert_in_darkmode" align=middle width=104.01404639999998pt height=22.465723500000017pt/> uniformly at
    random.

2.  Repeat the following and interrupt it when the total running time is
    more than $22.5\sqrt{N} + 1.4lg_2N$. Then go to stage 2(b)

    1.  Initialize the memory
        $\frac{1}{\sqrt{N}}\sum_j |j\rangle |y\rangle$. Mark every item
        j for which T\[j\]$<$T\[y\].

    2.  Apply the quantum exponential searching algorithm of ( M.
        Boyeret al, 1998)

    3.  Observe the first register: let $y'$ be the outcome. If
        T\[$y'$\]$<$T\[y\], then set the threshold index y to $y'$

3.  Return y

\*Due to the limited scope of this paper, the quantum exponential
algorithm has not been explored in depth.

### Exponential Speedup of SVM through Quantum Means

This method makes the use of Quantum Computing paradigms to speed up
classical operations to gain speed boosts. Through a series of
replacements of Classical methods of computation Quantum, different
processes such as finding the dot products, matrix inversions, etc speed
up SVM algorithm from <img src="https://rawgit.com/hurryingauto3/QuantumCompFinalProject/master/svgs/ea451baab95f872becd610647d21ca6e.svg?invert_in_darkmode" align=middle width=116.51134274999998pt height=26.76175259999998pt/> to O(log(MN)). This is achieved
through use of the following quantum operations

1.  Calculating the Kernal Matrix by Quantum Means (Suykens and
    Vandewalle, 1999)

2.  Quantum matrix inversion (Harrow et al., 2009)

3.  Simulation of sparse matrixes (Berry et al., 2007)

4.  Non-sparse density matrices reveal the eigenstructure exponentially
    faster than in classical algorithms (Lloyd et al., 2009);

A Comprehensive Model of QML
============================

In this section, this paper will attempt to create an overall
description of Quantum Machine Learning by making use of the results
that have been explored above.

As we have seen the aim of Quantum Machine Learning seems to be the same
as that of classical machine learning - train machines through data so
that they can make autonomous decisions. However, despite multiple
academic papers being published each day, the field of QML is fragmented
and still open to a lot of interpretation. [@BOOK:1] This may be owed to
the fact that machine learning itself is still a filed with ongoing
research. Moreover, there are a lot of factors that come into play when
developing a machine learning algorithm in the quantum paradigm.
Questions about the types of input and output, generalization
performance, and etc. Given below is a list of of Machine Learning
algorithms that have been researched in the quantum paradigm.

![Table of ML Algorithms [@BOOK:1]](images/table.jpg)

These differences in each algorithm keep us from generalizing the notion
of quantum machine learning. However they each seem to share one common
goal. QML aims to improve the current methodologies present in Machine
Learning for speeding up processes. These results are achieved through
various techniques that are common in Quantum Computing, within the
scope of this paper two techniques of Quantum Computing have been
applied to machine learning to provide a speed up to the machine
learning algorithms. [@article4]

#### 

**Quantum Approaches Discussed in this Paper**

1.  Use of Quantum Algorithms.

2.  Use of Enhancement of Classical Algorithms through Quantum means.

It is worth noting that only one of these applications show some
semblance of consistency as they seem to be less open ended problems
than the other. The second approach is seen to be only interested in
speeding up current algorithms by making use of the powerful
applications of quantum mechanics such as superposition, quantum
entanglement, etc. This approach uses Quantum Computing as an
enhancement to improve the learning times of machines. Then quantum
machine learning can be simplified by the diagram below.

![image](images/qmldiag.png)

Since, Quantum Algorithms work with qubits, the classical data must be
encoded for it to be able to undergo quantum state changes, and etc.
However, the final output will be classical data as that is what is
useful to us. The advantage here then becomes the speedup of the machine
learning process. The QML black box can contain any algorithm that uses
quantum properties and the general sketch of the overall structure of
quantum machine learning remains the same i.e. the inputs and outputs
are classical data. [@article4]

Conclusion
==========

Machine Learning is application of computers that has been around for a
decent amount of time and has made some decent strides in the last few
decades. Despite there still being ongoing research in the field, ML has
managed to integrate itself into many applications of our day to day
lives. However, as problems get more complication, and data sets get
larger classical machine learning seems to fall victim to the law of
diminishing returns. Quantum machine learning uses the paradigm of
quantum computing to attempt to combat these issues by making use of the
many algorithmic speed ups it has been able to achieve. QML is fairly
new and still a very open ended problem that can not be generalized
completely as of yet. Currently, most common quantum machine learning
solutions aim to provide a boost in speed up by enhancing classical
operations through quantum means, an example being SVM's that can
achieve an exponential speedup. The current model of quantum
enhancements converting classical data into quantum data for quantum
operations and the output is then converted back into classical data.
However, as research suggests this may only be a temporary generality
and is subject to change.
