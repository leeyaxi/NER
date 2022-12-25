### model list

1. aspect extraction
2. aspect-level polarity classification

### requirement

1. PyTorch == 1.8.0
2. transformers==4.18.0
3. nltk>=3.7
4. python3.6+

### 项目结构
```
├── datasets #数据集
│  └── MOOC
│      ├── mooc.test.txt.atepc
│      ├── mooc.train.txt.atepc
├── dependency #stanfor调用依赖
│  ├── DLUT_emotionontology.csv #知识图谱文件
│  ├── posguide.3rd.ch.pdf #词性说明文档
│  ├── stanford-corenlp-4.4.0-models-chinese.jar #中文模型
│  └── stanford-parser-full-2020-11-17 #parser调用java包
│      ├──...
│  └── stanford-postagger-full-2020-11-17 #词性分析调用java包
│      ├──...

├── losses #损失函数, 目前用到ce_loss, 有需求可以自行添加
│  ├── __init__.py
│  ├── ce_loss.py
│  ├── focal_loss.py
│  └── label_smoothing.py
├── metrics #评价指标, 可自定义,目前直接调用seqeval.metrics或者sklearn.metrics
│  ├── __init__.py
│  └── ner_metrics.py
├── models #模型构建 
│  ├── __init__.py
│  ├── bertApc.py #方面词抽取,命名实体识别任务
│  ├── MFSGC.py #方面词的polarity识别, 分类任务
│  └── layers #模型依赖的网络层, 可参考自定义自己的网络层
│      ├── __init__.py
│      ├── attention.py
│      ├── bilstm.py
│      ├── cnn.py
│      ├── crf.py
│      ├── dynamic_rnn.py
│      ├── embedding.py
│      └── linears.py
│      └── TransE.py
│      └── SGC.py
├── outputs #配置的输出路径
├── pretrain_model #配置的初始化模型路径(中文)
│  └── bert-base-chinese
│      ├── config.json
│      ├── pytorch_model.bin
│      ├── tokenizer.json
│      ├── tokenizer_config.json
│      └── vocab.txt
├── processors #文本预处理
│  ├── __init__.py
│  ├── apc_seq.py #方面词标注
│  ├── ape_seq.py #方面极性识别
│  └── utils_ner.py #预处理依赖函数
├── run_ap_class.py #方面词提取项目运行主函数
├── run_ap_polarity.py #方面词极性分析项目运行主函数
├── scripts
│  └── run_ap_class.sh #方面词提取项目运行的sh文件
│  └── run_ap_polarity.sh #方面词极性分析项目运行的sh文件

```


### 配置参数说明
  --model_type=bert #模型名称, 这里不用修改
  --model_name_or_path=$BERT_BASE_DIR #bert初始化模型路径, 可自行去huggface上下载对应语种的model
  --task_name=$TASK_NAME #任务名称, 可配置'**mooc**', 可自行增加task数据处理类
  --do_train #是否训练, 训练会包含验证过程, 每一轮会打印验证集效果并保存f1 best model
  --do_predict  #是否测试, 调用测试集测试
  --do_lower_case #文本大小写, 与根据初始化模型里的tokenizer配置保持一致
  --data_dir=$DATA_DIR/${TASK_NAME} #数据集路径名称
  --train_max_seq_length=128 #训练的max sequence length
  --eval_max_seq_length=128 #测试的max sequence length
  --per_gpu_train_batch_size=64 #训练的batch size
  --per_gpu_eval_batch_size=64  #测试的batch size
  --learning_rate=1e-5 #model学习率
  --crf_learning_rate=1e-3 #crf层学习率, 与model分开(仅在方面词提取阶段使用)
  --num_train_epochs=10  #训练轮次
  --logging_steps=-1 #日志等级, 不要修改
  --save_steps=-1 #每一轮训练途中是否调用验证过程, 不要修改
  --output_dir=$OUTPUR_DIR/${TASK_NAME}/ #model以及预测结果输出路径
  --overwrite_output_dir #是否需要覆盖输出路径
  --seed=42 #随机种子数

### 运行代码
#### Notice
1. linux下运行.sh文件scripts/run_ner_xxx.sh, pycharm下请自行配置环境, 参考run_ner_xxx.sh

2. Modify the configuration information in `run_ap_xxx.py` or `run_ap_xxx.sh` .

3. `sh scripts/run_ap_xxx.sh`