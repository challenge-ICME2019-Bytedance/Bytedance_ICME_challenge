REPO DESCRIPTION
-------  
Our FM implementation is based on tensorflow 1.12.0.  
You can run our reference training code on-the-fly using the following command:  

```
#--------------------------how to train----------------------------#
./train.sh <training_path> <validation_path> <save_model_dir> <batch_size> <embedding_size> \
           <optimizer [adagrad, adam]> <lr> <task [finish, like]> <track [1, 2]>  
```
 
CODE STRUCTURE
--------------  

<pre><code> 
#--------------------------run script------------------------------#
train.sh  

#----------------------------train---------------------------------#
train.py  

#------------------------common operation--------------------------#
common/  
  &nbsp; &nbsp; &nbsp; model_args.py  
  
#--------convert input text data into tensorflow batch need--------#
data_io/  
 &nbsp; &nbsp; &nbsp; data_parser.py  

#-------------prepare model and build up main framework------------#
models/  
 &nbsp; &nbsp;&nbsp;  model.py  

#---------------common algorithm and models for recom--------------#
model_zoo/  
 &nbsp; &nbsp; &nbsp; fm.py  

#-----------------utils for str or data processing-----------------#
utils/  
 &nbsp; &nbsp; &nbsp; utils.py
 </code></pre>


ALGORITHM: FACTORIZATION MACHINE
--------------------------------

![image](https://github.com/challenge-ICME2019-Bytedance/Bytedance_ICME_challenge/raw/master/images/fm.jpg)

BASELINE
-------  
Our baseline results with 5 features (user_id, user_city, item_id，author_id，item_city):   

  * TRACK2 LIKE TASK: 
  ```  
    auc: 86.5% 
    #------------------------params-------------------------#
    embedding_size = 40
    optimizer = adam  
    lr = 0.0005
  ```
  * TRACK FINISH TASK: 
  ```  
    auc: 69.8% 
    #------------------------params-------------------------#
    embedding_size = 40
    optimizer = adam   
    lr = 0.0001
  ```



