
## To use 1.x version of Tensorflow in Google Colab
```python
%tensorflow_version 1.x
```

## To mount Google drive in Google Colab
```python
from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive
%cd /gdrive/My Drive/xxx #xxx is your folder
%ls
```
