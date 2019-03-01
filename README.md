# AdaBound-Tensorflow
## In progress
Simple Tensorflow implementation of ["Adaptive Gradient Methods with Dynamic Bound of Learning Rate" (ICLR 2019)](https://openreview.net/forum?id=Bkg3g2R9FX)

## Hyperparameter
* `learning_rate` = 0.001
* `final_lr` = 0.01
* `beta1` = 0.9
* `beta2` = 0.999

## Usage
```python
  from AdaBound import AdaBoundOptimizer
  
  train_op = AdaBoundOptimizer(learning_rate=0.001, final_lr=0.01, beta1=0.9, beta2=0.999, amsgrad=False).minimize(loss)
```

## Network Architecture
```python
  x = fully_connected(inputs=images, units=100)
  x = relu(x)
  logits = fully_connected(inputs=x, units=10)
```

## Fashion-mnist Result
<div align="center">
   <img src="/assets/loss.png" width="420">
  <img src="/assets/acc.png"  width="420">
</div>

## Related works
* [AMSGrad-Tensorflow](https://github.com/taki0112/AMSGrad-Tensorflow)

## Author
Junho Kim
