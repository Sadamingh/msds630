# MSDS 630 Advanced Machine Learning University of San Francisco

----

## 1 Recommender System

### 1.1 Modeling Theory





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



* batch
* DataLoad





