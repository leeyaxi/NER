### dataset list

1. 数据获取: 采用Spider工具爬取ACL、EMNLP、NAACL的2018-2021年包含nlp关键词的论文,配置config.py, 运行spider.py;
2. 采用PyPDF2对pdf进行解析, 并采用spacy.tokenizer对文本进行word切分,最终获取7w条无监督样本
3. 用docker安装label-studio, 使用docker安装简单快捷, 并且该工具可以协同工作, 支持多人同时在线标注; 把标注的entity设置label-studio的项目中, 最终生成1k条标注样本
4. 对1k条标签数据进行8:2的数据切分, 前者作为训练集, 后者作为测试集
5. 在模型蒸馏时利用了除标注数据以外的无监督样本

### model list

1. 方案1: 模型采用bert+crf, 并用bert-base-uncased作为预训练模型在我们的任务上进行微调; 在小样本的背景下采用bert做encoder原因在于: 1、拥有大量的文本背景（预训练模型）2、优秀的表达能力: bert的attention机制能够更好的获取语义而不局限于样本规模
2. 方案2: 模型蒸馏，使用bert+crf作为teacher, 对训练集预测其soft label，student采用bilstm+crf, loss为teacher的hard label+student的soft label以及student的hard label。此外bilstm还可以继承bert的word embedding以及crf层。在小样本上直接采用bilstm+crf无法直接和bert相比,但采用蒸馏的方式可以在损失一点精度的前提下大大提高模型的效率
3. 方案3: 由于方案2中的训练集过少,直接蒸馏还是不理想,因此直接用bert+crf对7w条剩下的无监督预料进行打标作为hard label, bilstm+crf直接学习teacher的hard label, 在测试集上达到了75%的acc。结果显示还是大量的数据student才能学到teacher的label分布
具体论文可参考https://aclanthology.org/2021.naacl-main.454/

### experiments
方案1: 测试集结果
acc: 0.9704 - recall: 0.8945 - f1: 0.9309 - loss: 0.7765 
******* DatasetName results ********
acc: 0.9783 - recall: 0.7377 - f1: 0.8411 
******* HyperparameterName results ********
acc: 0.9540 - recall: 0.9540 - f1: 0.9540 
******* HyperparameterValue results ********
acc: 0.9375 - recall: 0.7895 - f1: 0.8571 
******* MethodName results ********
acc: 0.9740 - recall: 0.8772 - f1: 0.9231 
******* MetricName results ********
acc: 0.9808 - recall: 0.9808 - f1: 0.9808 
******* MetricValue results ********
acc: 0.8667 - recall: 0.7647 - f1: 0.8125 
******* TaskName results ********
acc: 0.9854 - recall: 0.9441 - f1: 0.9643
对测试集的acc、recall以及f1进行统计, acc虽然高但是recall值普遍偏低，尤其是在实体DatasetName、HyperparameterValue以及MetricValue, 
其原因应该在于训练集中标签比较少,其数据分布不够完全刻画其实体，因此虽然acc可以学到但是对于模型没有见过的分布时其效果就会比较差,因此需要用f1值
来刻画entity的最终performance。
loss分析：根据训练集和测试集的loss下降趋势来看，尽管受限数据集规模，但是模型依然可以学到实体的分布情况，因此训练集跟测试集的loss都
表现为正向

方案3: 测试集结果
acc: 0.7351 - recall: 0.5460 - f1: 0.6266
******* DatasetName results ********
acc: 0.5000 - recall: 0.0984 - f1: 0.1644
******* HyperparameterName results ********
acc: 0.7089 - recall: 0.6437 - f1: 0.6747
******* HyperparameterValue results ********
acc: 0.9231 - recall: 0.6316 - f1: 0.7500
******* MethodName results ********
acc: 0.7025 - recall: 0.5030 - f1: 0.5862
******* MetricName results ********
acc: 0.7059 - recall: 0.4615 - f1: 0.5581
******* MetricValue results ********
acc: 0.8125 - recall: 0.7647 - f1: 0.7879
******* TaskName results ********
acc: 0.7829 - recall: 0.7266 - f1: 0.7537
对测试集的acc、recall以及f1进行统计, bilstm在acc的效果依然比recall高，原因主要在于student依然学到的是teacher的分布, teacher在数据分布上
acc要高于recall，因此student保持该现象是在预期内的。此外对于teacher的recall值表现较差的实体DatasetName、HyperparameterValue以及MetricValue
上，除了HyperparameterValue在student上的效果比较好以外，DatasetName跟MetricValue都异常的差，尤其是前者，这说明teacher对DatasetName实体的
分布存在偏差，student放大了此偏差导致效果更差。并且过低的recall导致f1的效果跟acc的差距比较大，由此可见我们在评估数据时需要在多个维度衡量模型对数据的拟合成都
loss分析：bilstm的loss表现在于，训练集的loss下降程度明显快于测试集，这说明bilstm对teacher的拟合程度比较强，这也说明了teacher学习到的数据分布
基本比较简单，因此student的学习更加简单，但测试集的loss显然比teacher的下限高很多，这说明实际的数据分布与teacher的标签差异还是存在，并且此差异比较大。

### qualitative analysis
1、bert在小数据集上的拟合程度比较强, 但明显存在过拟合, 这是由于训练集规模导致数据的分布跟实际的实体分布存在差异，而模型无法统计出复杂的实体分布情况，
但这种效果也是符合预期的，我们相信随着数据集中实体的增加，模型对实体的召回率会更高，达到更优秀的f1
2、标签蒸馏可以在更小的model上拟合出deep model（teacher）的logit分布，这对实际的应用是非常有帮助的。
3、在目前的实验中我们蒸馏损失的精度比较大，这并不表示蒸馏算法的真实效果丢失的程度会这么大，实际上可以认为bilstm是可以学到bert的语义表示的，受限于
我们数据集的规模——teacher的logit分布比较简单，缺失了对真实数据的刻画，因此主要问题并不是student无法学习到teacher的数据分布。我们相信只要teacher
的表达能力足够强student model完全能够达到更高的performance


### model list

2. BERT+CRF (teacher)
3. IDCNN+CRF (student)
4. BILSTM+CRF (student)

### requirement

1. PyTorch == 1.8.0
2. transformers==4.18.0
4. python3.6+

### 运行代码
1. teacher在scripts/run_ner_xxx.sh
2. student模型配置文件在scripts/run_ner_xxx_for_student.sh, student配置中BERT_BASE_DIR要写teacher模型路径
3. Modify the configuration information in `run_ner_xxx.py` or `run_ner_xxx.sh` .
4. `sh scripts/run_ner_xxx.sh`

### 蒸馏方式-2 (比运行run_ner_xxx_for_student.sh方式效果要好--原因是训练集太少直接用训练集蒸馏效果会比较差)
1. 可以用大量无监爬取数据直接作为test.txt让teacher预测hard label
2. 将1得到的数据直接作为student的训练集, 采用teacher方式继续训练, student继承teacher的某些层like BERT_BASE_DIR要写teacher模型路径
4. Modify the configuration information in `run_ner_xxx.py` or `run_ner_xxx.sh` .
5. `sh scripts/run_ner_xxx.sh`

### 更改配置
1. TRAIN_TYPE  -- teacher or student
2. OUTPUR_DIR -- 输出路径
3. TASK_NAME -- 数据集名称

**note**: file structure of the model

```text
├── prev_trained_model
|  └── bert_base
|  |  └── pytorch_model.bin
|  |  └── config.json
|  |  └── vocab.txt
|  |  └── ......
```