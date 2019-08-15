## gym-env-racing-v-0
GYM environment for reinforcement learning (vehicle racing /posible medical topic/)

### Install:
```pip install -e gym-eliftech-racing-v-0```


### run manual driving
```python gym_eliftech_racing/envs/racing_simple.py```

### generate dataset
```python preprocess.py ```
#### run model train 
```python train.py``` 

### run models test
```python test_model.py```

### floyd
floyd run --env tensorflow-1.13 "python gym_eliftech_racing/reinforcement_learning/learning.py --weights "discrete_CarRacing-v0_weights.h5f""


floyd run "git clone https://github.com/keras-rl/keras-rl.git && cd keras-rl && python setup.py install && cd ../ && python gym_eliftech_racing/reinforcement_learning/learning.py --weights 'discrete_CarRacing-v0_weights.h5f'"

