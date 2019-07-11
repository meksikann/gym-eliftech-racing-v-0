## gym-env-racing-v-0
GYM environment for reinforcement learning (vehicle racing /posible medical topic/)


### Install:
```pip install -e gym-eliftech-racing-v-0```


#### run train
```
python gym_eliftech_racing/reinforcement_learning/learning.py --weights "discrete_CarRacing-v0_weights.h5f"```

### floyd
floyd run --env tensorflow-1.13 "python gym_eliftech_racing/reinforcement_learning/learning.py --weights "discrete_CarRacing-v0_weights.h5f""


floyd run "git clone https://github.com/keras-rl/keras-rl.git && cd keras-rl && python setup.py install && cd ../ && python gym_eliftech_racing/reinforcement_learning/learning.py --weights 'discrete_CarRacing-v0_weights.h5f'"
