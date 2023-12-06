# Chapter 11 - Virtual Screening Example

The rd_filters.py script mentioned in Chapter 11 can be found at https://github.com/PatWalters/rd_filters

## To open `*.ipynb` file in docker container

### run docker container with port forward
```bash
docker run --gpus all --rm -it -p 8888:8888 -v $PWD:/root/mydir/ deepchemio/deepchem:2.4.0
```

### install jupyterlab inside of container
```bash
pip install jupyterlab
```

### run jupyterlab
```bash
jupyter-lab --ip=0.0.0.0 --allow-root
```

then, you will see this msg.

```
No web browser found: could not locate runnable browser.
[C 2023-12-06 02:38:26.678 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-3497-open.html
    Or copy and paste one of these URLs:
        http://e745707b498e:8888/lab?token=3f0406df37b8790f1c244ae3062ec87d9570705c63a98cc3
     or http://127.0.0.1:8888/lab?token=3f0406df37b8790f1c244ae3062ec87d9570705c63a98cc3
```

click that link and you're web browser will be open.
