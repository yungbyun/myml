## a_simple_neuron_with_Keras @ Kaggle
>* https://www.kaggle.com/code/yungbyun/a-simple-neuron-with-keras?scriptVersionId=146661775

## To use 1.x version of Tensorflow in Google Colab
```python
%tensorflow_version 1.x
```

## To mount Google drive in Google Colab 
#### 프로그램에서 구글 드라이브에 있는 데이터를 접근하기 위한 코드
```python
from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive
%cd /gdrive/My Drive/xxx #xxx is your folder
%ls
```
