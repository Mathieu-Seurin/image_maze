# image_maze
Simple maze with augmented state to try RL agent



Step :
- Intall dependancies (python3.6, numpy, matplotlib, pytorch 0.3.0, keras).
  Although the code should work with newer pytorch, the "volatile" keyword has
  been removed and some parts will work less efficiently.
- Clone repository and move to newly created "image_maze" directory
- Optional (if next step fails because of imports): add src to path
  ```
  export "PYTHONPATH=$PYTHONPATH:/home/insert_your_path_here/image_maze/src/"
  ```
- Generate image datasets :
```
cd src/
python3 pretraining/preprocess.py
cd ..
```

You can now launch experiments either one at a time :
```
python3 main.py -config config/base.json
```

or in parallel and with a queue on (possibly several) GPU :

```
python3 main.py -config config/base.json
```

### NB : using the parallel launcher can cause material damage; please make sure you understand it before launching it.
