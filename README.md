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

### 1.2.5





