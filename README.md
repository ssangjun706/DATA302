# Overview
## GRU Model
![](assets/sgan-modified-no-pool.png)

## GRU with Image Pool
![](assets/sgan-modified-pool.png)

## Average running time
|     | LSTM          | GRU | GRU+POOL        |
| --- | ------------- | --- | --------------- |
| ETH | 3.87s(t=2000) |     | 4.17s(t=2000)   |
| SDD |               |     | 90.84s (t=1000) |
 

## Performance (ADE/FDE)
|     | LSTM       | GRU | GRU+POOL  |
| --- | ---------- | --- | --------- |
| ETH | 7.02/12.86 |     | 3.94/7.70 |
| SDD |            |     | 1.45/2.41 |
