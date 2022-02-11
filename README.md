# MSDS 630 Advanced Machine Learning University of San Francisco

----

## 1 Recommender System

## 1.0 Video and Paper Problems

### 1.0.1 Video Problems

Ref: https://www.youtube.com/watch?v=zzTbptEdKhY

* What the video is about?

Answer: How to recommend content to users in YouTube.

* Decribe a little bit about the initial neural network.

Answer: Inputs are embedded videos watched and embedded search tokens with geographic information. There are three ReLU layers with a KNN and a softmax in the end.

* What is the cold start problem in this video?

Answer: If you have a new video, there will not be much signals or information to train the neural networks.

* What is one solution of the cold start problem?

Answer: The topic of the video would help.

* How they modify the initial model to solve the cold start problem?

Answer: Adding topic vector to the input.

* What's the method they use to generate topic vector?

Answer: Deep CNN, which means to train on the video shots or audio.

* How to maintain a huge list of topics?

Answer: Use a structured knowledge graph.

* What are the components of triples?

Answer: Entity, property, value.

* What is the solution if the names are not exactly match?

Answer: Look at all the hyperlinks inside wikipedia and these always links to many different names.

* How to decide which topic is the central topic of a video?

Answer: Looks at co-occurrence of topics on wikipedia, so some topics get more votes.

* Describe the steps of entity linking using textual metadata.

Answer: mention detection, disambiguation, pruning.

### 1.0.2 Paper Problems

Ref: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf

* What are the steps for building this recommendation system?

```
candidate generate, scoring, reranking
```

* What are the two models they use in building this recommendation system?

```
candidate generation model, ranking model
```

* Why do we need a ranking model?

```
By using a ranking model, they reduced the number of the recommendations based on candidate generations from hundreds to dozens, which makes it easier for users to browse.
```

### 1.1 Modeling Theory Problems

* What is gradient descent?

Answer: gradient descent is a iterative optimization algorithm for finding the minimum of a function.

* Suppose we are considering a matrix factorization model with `n` users and `m` items, with a embedding size of `K`. Then how many parameters show we have in this model.

```
K * (m + n)
```

* Given the gradient descent equations of matrix factorization with MSE in algebra expressions.

<img src="./images/Screen Shot 2022-02-11 at 12.16.11 AM.png" alt="Screen Shot 2022-02-11 at 12.16.11 AM" style="zoom:40%;" />

* Given the gradient descent equations of matrix factorization with bias using MSE as the loss in algebra expressions.

<img src="./images/Screen Shot 2022-02-11 at 11.16.24 AM.png" alt="Screen Shot 2022-02-11 at 11.16.24 AM" style="zoom:50%;" />

<img src="./images/Screen Shot 2022-02-11 at 11.16.31 AM.png" alt="Screen Shot 2022-02-11 at 11.16.31 AM" style="zoom:50%;" />

* Given the gradient descent equations of matrix factorization with L2 regularization using MSE as the loss in algebra expressions.

<img src="./images/Screen Shot 2022-02-11 at 11.27.32 AM.png" alt="Screen Shot 2022-02-11 at 11.27.32 AM" style="zoom:60%;" />

* Given the gradient equations of matrix factorization in matrix expressions.

<img src="./images/Screen Shot 2022-02-11 at 12.16.36 AM.png" alt="Screen Shot 2022-02-11 at 12.16.36 AM" style="zoom:40%;" />

Where,

<img src="./images/Screen Shot 2022-02-11 at 2.23.42 AM.png" alt="Screen Shot 2022-02-11 at 2.23.42 AM" style="zoom:50%;" />

* Given a loss function used for optimizing matrix factorization.

<img src="./images/Screen Shot 2022-02-11 at 12.16.23 AM.png" alt="Screen Shot 2022-02-11 at 12.16.23 AM" style="zoom:40%;" />

* How would you use a content based recommendation system for online blogs?

Answer: Analyze content of the blogs, then use the similarity between blogs to recommend blogs similar to what a user likes.

* How would you use a collaborative filtering recommendation system for online blogs?

Answer: Use past user behaviours and similarities between users and items simultaneously to provide recommendations.

* Select "explicit feedback" (E) or "implicit feedback" (I) for the following user data.

```
rating
clicks
transactions
purchases
navgigation history
```

Solution:

```
rating                  E
clicks                  I
transactions            I
purchases               I
navgigation history     I            
```

* Explain the difference between implicit feedback or explicit feedback?

```
- Implicit rating: make inference from user's behavior
- Explicit rating: ask users to rate items
```

* What are the benefits and drawbacks for implicit feedbacks?

```
- Benefits:
	- available and easy to get
	
- Drawbacks:
	- no negative feedback
	- no preference level
```

* What are the benefits and drawbacks for explicit ratings?

```
- Benefits:
	- balanced pos and neg feedbacks
	- can have preference level
	- clear and direct
	
- Drawbacks:
	- biased data
	- sparse data
	- difficult to get
```

* Suppose we want to build a recommendation system based on clicks and the data of clicks are listed as follows,

```
user      item
0         1
0         0
1         1
1         4
2         3
2         2
3         3
```

What is the Utility matrix we can use for this task?

Solution:

<img src="./images/Screen Shot 2022-02-11 at 10.31.09 AM.png" alt="Screen Shot 2022-02-11 at 10.31.09 AM" style="zoom:50%;" />

* Following the last question, what is the utility matrix if we fill missing as negatives?

<img src="./images/Screen Shot 2022-02-11 at 10.41.59 AM.png" alt="Screen Shot 2022-02-11 at 10.41.59 AM" style="zoom:45%;" />

* Following the last question, what is the utility matrix if we use negative samples?

<img src="./images/Screen Shot 2022-02-11 at 10.44.16 AM.png" alt="Screen Shot 2022-02-11 at 10.44.16 AM" style="zoom:45%;" />



* Embedded user ratings are implicit feedbacks.

```
False
```

* What is the difference between model-based filtering and memory-based filtering?

Answer:

```
- Memory based: Remember the utility matrix and make recommendations based on the KNN algorithm. It tends to be slow. 
- Model based: Fit a model and make recommendations based on model predictions. It tends to have better performance. This is usually called matrix factorization.
```

* Suppose we want to predict the rating of user Alice on a movie by memory based collaborative filtering. Then what methods can we use?

```
- User-based KNN filtering: find users the nearnest to Alice, then make predictions based on the average of the ratings on that movie.
- Item-based KNN filtering: find the ratings of movies similar to that movie and then predict the average ratings based on these movies.
```

* What is the cold start problem?

Answer: If you have a new user or a new video, there will not be much signals or information to train the neural networks.

* What are the solutions for the user cold start problem?

Answer: recommend popular, require interests information, link to social media.

* How can we recommend based on the content of documents? What are the steps?

Answer: use TF-IDF. The steps are

```
- Remove stopwords
- Compute TF-IDF scores
- Keep works with high score
- Make a vector of size N to represent the document
```

* Given the following two users' ratings on items, calculate the Jaccard distance between them.

```
user 1        4 5   5 1   3 2
user 2          3 4 3 1 2 1
```

Solution:

```
sim = 1/8
```

* Given the following two users' ratings on items, and treating rating 1 and 2 to 0 and 3, 4, 5 to 1. Then calculate the Jaccard distance between them.

```
user 1        4 5   5 1   3 2
user 2          3 4 3 1 2 1
```

Solution:

```
user 1        1 1   1 0   1 0
user 2          1 1 1 0 0 0
```

If we fill missing with negatives,

```
user 1        1 1 0 1 0 0 1 0
user 2        0 1 1 1 0 0 0 0
```

Then,

```
sim = 5/8
```

* Given the following two users' ratings on items, calculate the cosine similarity between them.

```
user 1        1 5 3
user 2        1 2 3
```

Solution:

```
cosine sim = (1 + 10 + 9) / (sqrt(1 + 25 + 9) * sqrt(1 + 4 + 9))
           = 20 / (sqrt(35) * sqrt(14))
           = 20 / (7 * sqrt(10)) 
```

* Given the following two users' ratings on items, calculate the pearson similarity between them.

```
user 1        1 5 3
user 2        1 2 3
```

Solution:

```
avg_u1 = 3
avg_u2 = 2
pearson sim = (2 + 0 + 0) / (sqrt(4 + 4 + 0) + sqrt(1 + 0 + 1))
            = 2 / (3 * sqrt(2))
```

* What are the steps for content based kNN?

```
- Compute profile vectors for users and items
- Find a similarity measure and compute similarity between users and items
- Recommend to a user items with high similarity by kNN
```

* What are the benefits and drawbacks for a content based system?

```
Benefits:
	- no need data on other users
	- don't have a cold start problem
	- can be explained
	
Drawbacks:
	- difficult to construct feature vector
	- difficult to recognize new genres
	- hard to exploit the qualify of judgements from other users
```

### 1.2 Programming

### 1.2.1 Predictions

Suppose we are given the embedding matrices `emb_user` and `emb_movie` from fitting the matrix factorization problem and the model is,

```
y_hat = U * V.T
```

We also have a tabular dataset `df` which has the data as follows,

| userId | movieId | rating |
| :----: | :-----: | :----: |
|   0    |    0    |   4    |
|   0    |    1    |   3    |
|   1    |    1    |   5    |
|   1    |    2    |   5    |

Please write a function in `numpy` (alias `np`) and  `pandas` (alias `pd` ) to compute the prediction column without loops.

Solution:

```python
df['prediction'] = np.sum(emb_user[df['userId'].values]*emb_movie[df['moiveId'].values], axis=1)
```

### 1.2.2 Predictions

Suppose we are given the embedding matrices `emb_user` , `emb_movie` , `emb_user_bias` , `emb_movie_bias` from fitting the matrix factorization problem and the model is,

```
y_hat = U * V.T + B_U + B_V
```

We also have a tabular dataset `df` which has the data as follows,

| userId | movieId | rating |
| :----: | :-----: | :----: |
|   0    |    0    |   4    |
|   0    |    1    |   3    |
|   1    |    1    |   5    |
|   1    |    2    |   5    |

Please write a function in `numpy` (alias `np`) and  `pandas` (alias `pd` ) to compute the prediction column without loops.

Solution:

```python
df['prediction'] = np.sum(emb_user[df['userId'].values]*emb_movie[df['moiveId'].values], axis=1) + emb_user_bias[df['userId'].values] + emb_movie_bias[df['moiveId'].values]
```

### 1.2.3 Gradients

Suppose we are using MSE for gradient descent and we are given the embedding matrices `emb_user` and `emb_movie` from fitting the matrix factorization problem.

We also have a tabular dataset `df` which has the data as follows,

| userId | movieId | rating |
| :----: | :-----: | :----: |
|   0    |    0    |   4    |
|   0    |    1    |   3    |
|   1    |    1    |   5    |
|   1    |    2    |   5    |

 `Y` is a sparse representation of `df`, and we are given `Y_hat` which is a sparse matrix of the prediction if the items appears in `df`. 

Please write a function in `numpy` (alias `np`) and  `pandas` (alias `pd` ) to compute the gradients without loops.

Solution:

```
grad_user = -2 / N * np.dot((Y.toarray() - Y_hat.toarray()), emb_movie)
grad_movie = -2 / N * np.dot((Y.toarray() - Y_hat.toarray()), emb_user)
```

### 1.2.4 Gradient Descents

Suppose we are using momentum for calculating gradient descents and we are given the embedding matrices `emb_user` and `emb_movie` from fitting the matrix factorization problem. The dataset `df` and its sparse representation `Y` are provided. 

There are also some other variables we have defined,

- `v_user` : the history velocity of user embeddings for momentum gradient descent
- `v_movie` : the history velocity of movie embeddings for momentum gradient descent
- `beta` : the weight for historical velocity
- `iterations` : the numer of iterations for gradient descent
- `learning_rate` : the given learning rate

The gradient of user and movie and be calculated through calling `gradient` within each iteration.

```python
def gradient_descent(df, Y, emb_user, emb_movie):
  
    v_user = np.zeros_like(emb_user)
    v_movie = np.zeros_like(emb_movie)
    beta = 0.9
    iterations=100
    learning_rate=0.01

    for i in range(iterations):
      	grad_user, grad_movie = gradient(df, Y, emb_user, emb_movie)
        # TODO: Implement here.
        
        
    return emb_user, emb_movie
```

Please implement the function in `numpy` (alias `np`) and  `pandas` (alias `pd` ) to compute the gradient descents.

Solution: 

```python
v_user = beta * v_user + (1 - beta) * grad_user
v_movie = beta * v_movie + (1 - beta) * grad_movie

emb_user -= learning_rate * v_user
emb_movie -= learning_rate * v_movie
```



## 2 Basic PyTorch

### 2.1 Non-Programming Question

* What is the shape of the tensor `out`?

```python
embed = nn.Embedding(5,7)
x = torch.LongTensor([[1,0,1,4,2]])
out = embed(x)
```

Answer:

```
[1, 5, 7]
```

* What is the minimum valid integer for the blank?

```python
embed = nn.Embedding(_______,7)
x = torch.LongTensor([[1,1,2,2,3],[5,4,4,2,1]])
out = embed(x)
```

Answer:

```
6
```

* What is the shape of the tensor `out`?

```python
embed = nn.Embedding(10,5)
x = torch.LongTensor([1,0,1,4,2,0])
out = embed(x)
```

Answer:

```
[6, 5]
```

* What is the value of `result` in the last line?

```python
embed = nn.Embedding(10,5)
x = torch.LongTensor([1,0,1,4,2,1])
out = embed(x)
bools = out[0] == out[2]
result = bools.detach().numpy()[0]
```

Answer:

```
True
```

* What is the value of the tensor *x.grad* after running the next few lines?

```
x = torch.tensor([1.0,3.0,2.0], requires_grad=True)
L = (3*x + 7).sum()
L.backward()
```

Answer:

```
tensor([3., 3., 3.])
```

* What is the value of the tensor *x.grad* after running the next few lines?

```
x = torch.tensor([1.0,3.0,2.0], requires_grad=True)
L = (-2*x**2 + 3*x + 7).mean()
L.backward()
```

Answer:

```
tensor([-0.3333, -3.0000, -1.6667])
```

* How to create a tensor of zeros shaped `[2,2,2]`?

```
torch.zeros(2,2,2)
```

* How to create a tensor of ones shaped `[2,2,2]`?

```
torch.ones(2,2,2)
```

* How to create a tensor of normal distributed random values ranged [0, 1) of shape `[2,2,2]`?

```
torch.rand(2,2,2)
```

* How to create a tensor of uniform distributed random integers ranged [3, 10) of shape `[2,2,2]`?

```
torch.randint(3, 10, (2,2,2))
```

* What is the shape of the `x` after the following lines?

```
x = torch.tensor([[1,2,3,4]])
x = x.squeeze()
```

Answer:

```
[4]
```

* What is the shape of the `x` after the following lines?

```
x = torch.tensor([[1,2,3,4]])
x = x.squeeze(1)
```

Answer:

```
[1, 4]
```

* What is the shape of the `x` after the following lines?

```
x = torch.tensor([[1],[2],[3],[4]])
x = x.squeeze()
```

Answer:

```
[4]
```

* What is the shape of the `x` after the following lines?

```
x = torch.tensor([[1,2,3,4]])
x = x.unsqueeze(1)
```

Answer:

```
[1,1,4]
```

* What is the shape of the `x` after the following lines?

```
x = torch.tensor([[1],[2],[3],[4]])
x = x.unsqueeze(1)
```

Answer:

```
[4,1,1]
```

* What is the shape of the `x` after the following lines?

```
x = torch.tensor([1,2,3,4])
x = x.unsqueeze(1)
```

Answer:

```
[4,1]
```

* Describe what each of the following lines of code are doing during a training loop.

```
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Answer:

```
line 1: zero out the gradients for all the tensors in the model
line 2: computing the gradients for all tensors based on loss
line 3: iterate all thee tensors and use the interally stored grad to update their values
```

* Suppose we are given batch size of 1,000 and the data size is 1,000,000, then calculate how many batches do we have in one epoch.

```
1000000/1000 = 1000
```

* Suppose we are given batch size of 1,000 and the data size is 1,000,000. The total number of epoches is 10 per training. Calculate the number of iterations we have in one training.

```
1000000/1000 * 10 = 10000
```



### 2.2 Programming Question

### 2.2.1 PyTorch Model

Suppose we have a linear regression problem with 4 input variables. Write a model in PyTorch that can be used for this task. Suppose we have imported,

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Use the following template for your implementation.

```python
class LinearRegression(torch.nn.Module):
  
    def __init__(self):
        super(LinearRegression, self).__init__()
        # TODO: Implement this method
        
    def forward(self, x):
        # TODO: Implement this method
      
model = LinearRegression()
```

Solution:

```python
class LinearRegression(torch.nn.Module):
  
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(4, 1)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
      
model = LinearRegression()
```

### 2.2.2 PyTorch Model

Suppose we have a logistic regression problem with 4 input variables. Write a model in PyTorch that can be used for this task. Suppose we have imported,

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Use the following template for your implementation.

```python
class LogisticLinearRegression(torch.nn.Module):
  
    def __init__(self):
        super(LinearRegression, self).__init__()
        # TODO: Implement this method
        
    def forward(self, x):
        # TODO: Implement this method
      
model = LogisticLinearRegression()
```

Solution:

```python
class LogisticRegression(torch.nn.Module):
  
     def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(4, 1)
        
     def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
      
model = LogisticRegression()
```

### 2.2.3 PyTorch Model

Suppose we have a classification problem with 10 input variables and 3 output classes. Write a model in PyTorch that can be used for this task. Suppose we have imported,

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Use the following template for your implementation.

```python
class Classification(torch.nn.Module):
  
    def __init__(self):
        super(Classification, self).__init__()
        # TODO: Implement this method
        
    def forward(self, x):
        # TODO: Implement this method
      
model = Classification()
```

Solution:

```python
class Classification(torch.nn.Module):
  
    def __init__(self):
        super(Classification, self).__init__()
        self.linear = torch.nn.Linear(10, 3)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        y_pred = self.softmax(self.linear(x))
        return y_pred
      
model = Classification()
```

### 2.2.4 PyTorch Model

Suppose we have a matrix factorization problem with embedding size of 100 without bias. Write a model in PyTorch that can be used for this task. Suppose we have imported,

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Use the following template for your implementation.

```python
class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        # TODO: Implement this method

    def forward(self, u, v):
        # TODO: Implement this method
        
model = MF()
```

Solution:

```python
class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)

    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        return (U * V).sum(1)
        
model = MF()
```

### 2.2.5 PyTorch Model

Suppose we have a matrix factorization problem with embedding size of 100 with bias. Write a model in PyTorch that can be used for this task. Suppose we have imported,

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Use the following template for your implementation.

```python
class BiasedMF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(BiasedMF, self).__init__()
        # TODO: Implement this method

    def forward(self, u, v):
        # TODO: Implement this method
        
model = BiasedMF()
```

Solution:

```python
class BiasedMF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(BiasedMF, self).__init__()
        
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_item, 1)
        
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        return (U * V).sum(1) + b_u + b_v
        
model = BiasedMF()
```

### 2.2.6 Train loop

Write a training loop in PyTorch that can train a linear regression model. The template is given as follows,

```python
def train_model(model, optimizer, train_dl, epoch=10):
	  for i in range(epochs):
    		model.train()
    		total = 0
        sum_loss = 0
        for x, y in train_dl:
          	batch = y.shape[0]
          	# TODO: Implement here.
          
          	total += batch
          	sum_loss += batch*(loss.item())
      	train_loss = sum_loss / total
        print(train_loss)
```

Solution:

```python
y_hat = model(x)
loss = F.mse_loss(y_hat, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 2.2.7 Train loop

Write a training loop in PyTorch that can train a logistic regression model (assume the model is linear and doesn't apply sigmoid in the end). The template is given as follows,

```python
def train_model(model, optimizer, train_dl, epoch=10):
	  for i in range(epochs):
    		model.train()
    		total = 0
        sum_loss = 0
        for x, y in train_dl:
          	batch = y.shape[0]
          	# TODO: Implement here.
          
          	total += batch
          	sum_loss += batch*(loss.item())
      	train_loss = sum_loss / total
        print(train_loss)
```

Solution:

```python
y_hat = model(x)
loss = F.binary_cross_entropy_with_logit(y_hat, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 2.2.8 Train loop

Write a training loop in PyTorch that can train a multiclass classification model. The template is given as follows,

```python
def train_model(model, optimizer, train_dl, epoch=10):
	  for i in range(epochs):
    		model.train()
    		total = 0
        sum_loss = 0
        for x, y in train_dl:
          	batch = y.shape[0]
          	# TODO: Implement here.
          
          	total += batch
          	sum_loss += batch*(loss.item())
      	train_loss = sum_loss / total
        print(train_loss)
```

Solution:

```
y_hat = model(x)
loss = F.cross_entropy(y_hat, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 2.2.9 Dataset & Dataloader

Suppose we are given `x_train` and `y_train` , construct a dataloader `train_dl` with batch size of 1000, and shuffle the data.

Solution:

```python
class TrainingDataset(data_utils.Dataset):

    def __init__(self, x, y):
        x = torch.tensor(x).float()
        y = torch.tensor(y).float().unsqueeze(1)
        self.x, self.y = x, y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
      
train_ds = TrainingDataset(x_train, y_train)
train_dl = data_utils.DataLoader(train_ds, batch_size=1000, shuffle=True)
```

### 2.2.10 Dataloader

Suppose we want to get one batch of data from the dataloader `train_dl`. Please give the function for this task.

Solution:

```python
x, y = next(iter(train_dl))
```

### 2.2.11 Dataloader

Write a function that given a model `model` and a dataloader  `train_dl` that computes the balanced accuracy. Assume that you have a binary classification problem and you can use `balanced_accuracy_score` from `sklearn`.

Solution:

```python
def metric_accuracy(model, dataload):

    model.eval()
    accuracies = []
    
    for x, y in dataload:
      
        y_hat = model(x)
        accuracy = sklearn.metrics.balanced_accuracy_score(y, y_hat)
        accuracies.append(accuracy)
        
    return np.mean(accuracies)

metric_accuracy(model, train_dl)
```

