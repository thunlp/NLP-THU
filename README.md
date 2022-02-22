This repository provides reading materials recommended by NLP-THU Course.

## 1. Introduction
### Introduction
1. **Foundations of statistical natural language processing**. Christopher D. Manning and Hinrich Schütze. MIT Press 2001. [[link]](https://www.cs.vassar.edu/~cs366/docs/Manning_Schuetze_StatisticalNLP.pdf)
2. **Introduction to information retrieval**. Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze. Cambridge University Press 2008. [[link]](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)
3. **Semantic Relations Between Nominals**. Vivi Nastase, Preslav Nakov, Diarmuid Ó Séaghdha and Stan Szpakowicz. Morgan & Claypool Publishers 2013. [[link]](https://www.morganclaypool.com/doi/abs/10.2200/S00489ED1V01Y201303HLT019)

## 2. Word Representation and Neural Networks
### a. Word Representation
1. **Linguistic Regularities in Continuous Space Word Representations**. Tomas Mikolov, Wen-tau Yih and Geoffrey Zweig. NAACL 2013. [[link]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf)
2. **Glove: Global Vectors for Word Representation**. Jeffrey Pennington, Richard Socher and Christopher D. Manning. EMNLP 2014. [[link]](https://nlp.stanford.edu/pubs/glove.pdf)
3. **Deep Contextualized Word Representations**. Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee and Luke Zettlemoyer. NAACL 2018. [[link]](https://arxiv.org/pdf/1802.05365.pdf)
4. **Parallel Distributed Processing**. Jerome A. Feldman, Patrick J. Hayes and David E. Rumelhart. 1986.

### b. RNN & CNN
1. **ImageNet Classification with Deep Convolutional Neural Networks**. NIPS 2012  [[link]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
2. **Convolutional Neural Networks for Sentence Classification**. EMNLP 2014 [[link]](https://www.aclweb.org/anthology/D14-1181.pdf)
3. **Long short-term memory**. MIT Press 1997 [[link]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)



## 3. Seq2Seq Modeling
### a. Machine Translation
#### Must-read Papers
1. **The Mathematics of Statistical Machine Translation: Parameter Estimation**. Peter EBrown, Stephen ADella Pietra, Vincent JDella Pietra, and Robert LMercer. Computational Linguistics 1993 [[link]](http://aclweb.org/anthology/J93-2003)
2. **(Seq2seq) Sequence to Sequence Learning with Neural Networks**. Ilya Sutskever, Oriol Vinyals, and Quoc VLe. NIPS 2014 [[link]](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
3. **(BLEU) BLEU: a Method for Automatic Evaluation of Machine Translation**. Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. ACL 2002 [[link]](http://aclweb.org/anthology/P02-1040)
#### Further Reading
1. **Statistical Phrase-Based Translation**. Philipp Koehn, Franz JOch, and Daniel Marcu. NAACL 2003 [[link]](http://aclweb.org/anthology/N03-1017)
2. **Hierarchical Phrase-Based Translation**. David Chiang. Computational Linguistics 2007  [[link]](http://aclweb.org/anthology/J07-2003)
3. **(Beam Search) Beam Search Strategies for Neural Machine Translation**. Markus Freitag and Yaser Al-Onaizan. 2017 [[link]](http://aclweb.org/anthology/W17-3207)
4. **MT paper list**.  [[link]](https://github.com/THUNLP-MT/MT-Reading-List)
5. **THUMT toolkit**.  [[link]](https://github.com/THUNLP-MT/THUMT)
### b. Attention
1. **Introduction to attention**.  [[link]](https://ruder.io/deep-learning-nlp-best-practices/index.html#attention)
2. **Neural Machine Translation by Jointly Learning to Align and Translate**. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. ICLR 2015 [[link]](https://arxiv.org/pdf/1409.0473.pdf)
### c. Transformer
#### Must-read Papers
1. **(Transformer) Attention is All You Need**. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan NGomez, Lukasz Kaiser, and Illia Polosukhin. NIPS 2017 [[link]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
2. **(BPE) Neural Machine Translation of Rare Words with Subword Units**. Rico Sennrich, Barry Haddow, and Alexandra Birch. ACL 2016 [[link]](https://arxiv.org/pdf/1508.07909.pdf)
#### Further Reading
1. **Illustrated Transformer**.  [[link]](http://jalammar.github.io/illustrated-transformer/)
2. **Layer normalization. Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E**. Hinton. 2016 [[link]](https://arxiv.org/abs/1607.06450)
3. **Deep residual learning for image recognition**. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. CVPR 2016 [[link]](https://arxiv.org/abs/1512.03385)

## 4. Pre-Trained Language Models
### Must-read papers
1. **Semi-supervised Sequence Learning**.  [[link]](https://arxiv.org/pdf/1511.01432.pdf)
2. **(ELMo) Deep contextualized word representations**.  [[link]](https://arxiv.org/pdf/1802.05365.pdf)
3. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**.  [[link]](https://arxiv.org/pdf/1810.04805)
### Further Reading
1. **Introduction of Pre-trained LM**.  [[link]](https://arxiv.org/pdf/1810.04805.pdf)
2. **Transformer code repo**.  [[link]](https://github.com/huggingface/transformers)
3. **Transfer Learning in Natural Language Processing**. Sebastian Ruder, Matthew E. Peters, Swabha Swayamdipta, Thomas Wolf. NAACL 2019 [[link]](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit?usp=sharing)
4. **PLM paper list**.  [[link]](https://github.com/thunlp/PLMpapers)

## 5. Knowledge Graph
### a. Introduction to KG
1. **Towards a Definition of Knowledge Graphs**. Lisa Ehrlinger, Wolfram Wöß [[link]](https://www.researchgate.net/profile/Wolfram_Woess/publication/323316736_Towards_a_Definition_of_Knowledge_Graphs/links/5a8d6e8f0f7e9b27c5b4b1c3/Towards-a-Definition-of-Knowledge-Graphs.pdf)
2. **KG Definition & History Wiki** [[link]](https://en.wikipedia.org/wiki/Knowledge_Graph)
3. **Semantic Network** [[link]](https://en.wikipedia.org/wiki/Semantic_network)
### b. Knowledge Representation Learning
#### Must-read papers
1. **KRL paper list** [[link]](https://github.com/thunlp/KRLPapers)
2. **Knowledge Representation Learning: A Review**. (In Chinese) Zhiyuan Liu, Maosong Sun, Yankai Lin, Ruobing Xie. 计算机研究与发展 2016.  [[link]](http://crad.ict.ac.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=3099)
3. **A Review of Relational Machine Learning for Knowledge Graphs**. Maximilian Nickel, Kevin Murphy, Volker Tresp, Evgeniy Gabrilovich. 2016.  [[link]](https://arxiv.org/pdf/1503.00759.pdf)
4. **Knowledge Graph Embedding: A Survey of Approaches and Applications**. Quan Wang, Zhendong Mao, Bin Wang, Li Guo. TKDE 2017.  [[link]](http://ieeexplore.ieee.org/abstract/document/8047276/)
#### Further reading
1. **OpenKE** [[link]](http://openke.thunlp.org/)
### c. Reasoning
1. **KG Reasoning paper list** [[link]](https://github.com/THU-KEG/Knowledge_Graph_Reasoning_Papers) & PPT  [[link]](https://sites.cs.ucsb.edu/~william/papers/Part3_KB_Reasoning.pdf)

## 6. Information Extraction - 1 
### a. Part of Speech Tagging (POS Tagging)
1. **Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss**. Plank Barbara, Anders Søgaard, Yoav Goldberg. ACL 2016. [[link]](https://arxiv.org/pdf/1604.05529)
2. **Blog: NLP Guide: Identifying Part of Speech Tags using Conditional Random Fields**. [[link]](https://medium.com/analytics-vidhya/pos-tagging-using-conditional-random-fields-92077e5eaa31)

### b. Sequence Labelling
1. **Hierarchically-Refined Label Attention Network for Sequence Labeling**. Cui Leyang, Yue Zhang. EMNLP-IJCNLP 2019. [[link]](https://arxiv.org/pdf/1908.08676)
2. **End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF**. Ma Xuezhe, Eduard Hovy. ACL 2016. [[link]](https://arxiv.org/pdf/1603.01354)
3. **Comparisons of Sequence Labeling Algorithms and Extensions**. Nguyen Nam, Yunsong Guo. ICML 2007.  [[link]](http://www.cs.cornell.edu/~nhnguyen/icml07structured.pdf)
### c. Named Entity Recognition
1. **Blog: Named Entity Recognition Tagging**. CS230  [[link]](https://cs230.stanford.edu/blog/namedentity/)
2. **A Survey of Named Entity Recognition and Classification**. David Nadeau, Satoshi Sekine. Computational Linguistics 2007.  [[link]](https://nlp.cs.nyu.edu/sekine/papers/li07.pdf)
3. **Neural Architectures for Named Entity Recognition**. Lample Guillaume, et al. NAACL 2016. [[link]](https://pdfs.semanticscholar.org/0891/ed6ed64fb461bc03557b28c686f87d880c9a.pdf)
4. **Named Entity Recognition with Bidirectional LSTM-CNNs**. Jason P. C. Chiu, Eric Nichols. TACL 2016. [[link]](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00104)

## 7. Information Extraction - 2
### a. Relation Extraction
#### Must-read papers
1. **Relation Classification via Convolutional Deep Neural Network**. Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, Jun Zhao. COLING 2014.  [[link]](http://www.aclweb.org/anthology/C14-1220)
2. **Distant Supervision for Relation Extraction without Labeled Data**. Mike Mintz, Steven Bills, Rion Snow, Dan Jurafsky. ACL-IJCNLP 2009. [[link]](https://www.aclweb.org/anthology/P09-1113)
3. **Neural Relation Extraction with Selective Attention over Instances**. Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, Maosong Sun. ACL 2016. [[link]](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/acl2016_nre.pdf)
#### Further reading
1. **RE paper list** [[link]](https://github.com/thunlp/NREPapers)
### b. Advanced Topics
#### - Event Extraction
1. **Joint Event Extraction via Structured Prediction with Global Features**. Qi Li, Heng Ji and Liang Huang. ACL 2013. [[link]](https://www.aclweb.org/anthology/P13-1008/)
2. **Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks**. Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng and Jun Zhao. ACL 2015. [[link]](https://www.aclweb.org/anthology/P15-1017/)
3. **Adversarial Training for Weakly Supervised Event Detection**. Xiaozhi Wang, Xu Han, Zhiyuan Liu, Maosong Sun and Peng Li. NAACL 2019. [[link]](https://www.aclweb.org/anthology/N19-1105/)
4. **CLEVE: Contrastive Pre-training for Event Extraction**. Ziqi Wang, Xiaozhi Wang, Xu Han, Yankai Lin, Lei Hou, Zhiyuan Liu, Peng Li, Juanzi Li, Jie Zhou. ACL 2021. [[link]](https://bakser.github.io/files/ACL21-CLEVE/CLEVE.pdf)
#### - OpenRE 
1. **Open Relation Extraction: Relational Knowledge Transfer from Supervised Data to Unsupervised Data**. Ruidong Wu, Yuan Yao, Xu Han, Ruobing Xie, Zhiyuan Liu, Fen Lin, Leyu Lin, Maosong Sun. EMNLP 2019. [[link]](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/emnlp2019_rsn.pdf)
2. **Discrete-state Variational Autoencoders for Joint Discovery and Factorization of Relations**. Diego Marcheggiani and Ivan Titov. TACL 2016. [[link]](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiumpKG_8XnAhURPewKHS0bA_YQFjAAegQIBhAB&url=https%3A%2F%2Fwww.aclweb.org%2Fanthology%2FQ16-1017&usg=AOvVaw00Nk3Dzf54_rjbUDSpfjUC)
3. **Open Hierarchical Relation Extraction**. Kai Zhang, Yuan Yao, Ruobing Xie, Xu Han, Zhiyuan Liu, Fen Lin, Leyu Lin, Maosong Sun. NAACL 2021. [[link]](https://aclanthology.org/2021.naacl-main.452/)
#### - Document-Level RE
1. **DocRED: A Large-Scale Document-Level Relation Extraction Dataset**. Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin, Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou, Maosong Sun. ACL 2019. [[link]](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/acl2019_docred.pdf)
2. **A Walk-based Model on Entity Graphs for Relation Extraction**. Fenia Christopoulou, Makoto Miwa, Sophia Ananiadou. ACL 2017. [[link]](https://www.aclweb.org/anthology/P18-2014.pdf)
3. **Graph Neural Networks with Generated Parameters for Relation Extraction**. Hao Zhu, Yankai Lin, Zhiyuan Liu, Jie Fu, Tat-Seng Chua, Maosong Sun. ACL 2019. [[link]](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/acl2019_gnn4nre.pdf)
4. **Reasoning with Latent Structure Refinement for Document-Level Relation Extraction**. Guoshun Nan, Zhijiang Guo, Ivan Sekulić, Wei Lu. ACL 2020. [[link]](https://arxiv.org/abs/2005.06312)
#### - Few-shot RE
1. **FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation**. Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, Maosong Sun. ACL 2019 [[link]](https://www.aclweb.org/anthology/D18-1514.pdf)
2. **Matching Networks for One Shot Learning**. Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra [[link]](https://arxiv.org/abs/1606.04080)
3. **Prototypical Networks for Few-shot Learning**. Jake Snell, Kevin Swersky, Richard SZemel [[link]](https://arxiv.org/abs/1703.05175)
4. **Meta-Information Guided Meta-Learning for Few-Shot Relation Classification**. Bowen Dong, Yuan Yao, Ruobing Xie, Tianyu Gao, Xu Han, Zhiyuan Liu, Fen Lin, Leyu Lin, Maosong Sun. COLING 2020. [[link]](https://aclanthology.org/2020.coling-main.140/)

## 8. Knowledge-Guided NLP
#### Must-read papers
1. **ERNIE: Enhanced Language Representation with Informative Entities** Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, Qun Liu. ACL 2019 [[link]](https://www.aclweb.org/anthology/P19-1139.pdf)
2. **Neural natural language inference models enhanced with external knowledge**. Qian Chen, Xiaodan Zhu, Zhen-Hua Ling, Diana Inkpen, and Si Wei. ACL 2018 [[link]](https://www.aclweb.org/anthology/P18-1224)
3. **Neural knowledge acquisition via mutual attention between knowledge graph and text**. Xu Han, Zhiyuan Liu, and Maosong Sun. AAAI 2018 [[link]](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2018_jointnre.pdf)
#### Further reading
1. **Language Models as Knowledge Bases?** [[link]](https://www.aclweb.org/anthology/D19-1250.pdf)
2. **Knowledge enhanced contextual word representations**. Matthew EPeters, Mark Neumann, Robert Logan, Roy Schwartz, Vidur Joshi, Sameer Singh, and Noah ASmith. EMNLP 2019 [[link]](https://doi.org/10.18653/v1/D19-1005)
3. **Barack’s wife hillary: Using knowledge graphs for fact-aware language modeling**. Robert Logan, Nelson FLiu, Matthew EPeters, Matt Gardner, and Sameer Singh. ACL 2019 [[link]](https://doi.org/10.18653/v1/P19-1598)
4. **Knowledgeable Reader: Enhancing Cloze-style Reading Comprehension with External Commonsense Knowledge**. Todor Mihaylov and Anette Frank. ACL 2018 [[link]](https://www.aclweb.org/anthology/P18-1076)
5. **Improving question answering by commonsense-based pre-training**. Wanjun Zhong, Duyu Tang, Nan Duan, Ming Zhou, Jiahai Wang, and Jian Yin. 2018 [[link]](https://arxiv.org/pdf/1809.03568.pdf)
6. **Adaptive knowledge sharing in multi-task learning: Improving low-resource neural machine translation**. Poorya Zaremoodi, Wray Buntine, and Gholamreza Haffari. ACL 2018 [[link]](https://www.aclweb.org/anthology/P18-2104)

## 9. Advanced Learning Methods
#### a. Adversarial Training
##### Must-read papers
1. **Explaining and Harnessing Adversarial Examples**. Ian JGoodfellow, Jonathon Shlens, and Christian Szegedy. ICLR 2015 [[link]](https://arxiv.org/pdf/1412.6572.pdf))
2. **Generative Adversarial Nets**. Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. NIPS 2015 [[link]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
3. **Wasserstein GAN**. Martín Arjovsky, Soumith Chintala, and Léon Bottou. ICML 2017 [[link]](https://arxiv.org/abs/1701.07875)
##### Further reading
1. **Adversarial Examples for Evaluating Reading Comprehension Systems**. Robin Jia, Percy Liang. EMNLP 2017 [[link]](https://arxiv.org/pdf/1707.07328.pdf)
2. **Certified Defenses Against Adversarial Examples**. Raghunathan, Aditi, Jacob Steinhardt, and Percy Liang. ICLR 2018 [[link]](https://arxiv.org/pdf/1801.09344.pdf)
3. **Robust Neural Machine Translation with Doubly Adversarial Inputs**. Yong Cheng, Lu Jiang, and Wolfgang Macherey. ACL 2019 [[link]](https://www.aclweb.org/anthology/P19-1425.pdf)
4. **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**. Radford Alec, Metz Luke, and Chintala SoumithICLR 2016 [[link]](https://arxiv.org/pdf/1511.06434.pdf)
5. **Improved Training of Wasserstein GANs**. Martin Arjovsky, Soumith Chintala, and Léon BottouGulrajani Ishaan, Ahmed Faruk, Arjovsky Martin, Dumoulin Vincent, and Courville Aaron. NIPS 2017 [[link]](https://arxiv.org/abs/1704.00028)
6. **Are GANs Created Equal? A Large-scale Study**. Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, and Olivier Bousquet. NIPS 2018 [[link]](https://arxiv.org/pdf/1711.10337.pdf)
7. **Unsupervised Machine Translation Using Monolingual Corpora Only**. Guillaume Lample, Alexis Conneau, Ludovic Denoyer, and Marc'Aurelio Ranzato. ICLR 2018 [[link]](https://arxiv.org/pdf/1711.00043.pdf;Guillaume)
8. **Adversarial Multi-task Learning for Text Classification**. Pengfei Liu, Xipeng Qiu, and Xuanjing Huang. ACL 2017 [[link]](https://arxiv.org/pdf/1704.05742.pdf)
9. **SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient**. Lantao Yu, Weinan Zhang, Jun Wang, and Yong Yu. AAAI 2018 [[link]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14344/14489)
#### b. Reinforcement Learning
##### Must-read papers
1. **Playing atari with deep reinforcement learning**. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller. 2013 [[link]](https://arxiv.org/pdf/1312.5602.pdf)
2. **Human-level control through deep reinforcement learning**. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis. Nature 2015 [[link]](https://www.nature.com/articles/nature14236.pdf)
3. **Mastering the game of go with deep neural networks and tree search**. David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, Demis Hassabis. Nature 2016 [[link]](https://www.nature.com/articles/nature16961.pdf)
4. **Reinforcement learning for relation classification from noisy data**. Jun Feng, Minlie Huang, Li Zhao, Yang Yang, Xiaoyan Zhu. AAAI 2018 [[link]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/AAAI2018Denoising.pdf)
##### Further reading
1. **Reinforced co-training**. Jiawei Wu, Lei Li, William Yang Wang. NAACL 2018 [[link]](http://arxiv.org/pdf/1804.06035.pdf)
2. **Playing 20 question game with policy-based reinforcement learning**. Huang Hu, Xianchao Wu, Bingfeng Luo, Chongyang Tao, Can Xu, Wei Wu, Zhan Chen. EMNLP 2018 [[link]](https://www.aclweb.org/anthology/D18-1361.pdf)
3. **Entity-relation extraction as multi-turn question answering**. Xiaoya Li, Fan Yin, Zijun Sun, Xiayu Li, Arianna Yuan, Duo Chai, Mingxin Zhou, Jiwei Li. ACL 2019 [[link]](https://www.aclweb.org/anthology/P19-1129.pdf)
4. **Language understanding for text-based games using deep reinforcement learning**. Karthik Narasimhan, Tejas D Kulkarni, Regina Barzilay. EMNLP 2015 [[link]](https://www.aclweb.org/anthology/D15-1001.pdf)
5. **Deep reinforcement learning with a natural language action space**. Ji He, Jianshu Chen, Xiaodong He, Jianfeng Gao, Lihong Li, Li Deng, Mari Ostendorf. ACL 2016 [[link]](https://www.aclweb.org/anthology/P16-1153.pdf)
#### c. Few-Shot Learning
##### Must-read papers
1. **FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation**. Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, Maosong Sun. ACL 2019 [[link]](https://www.aclweb.org/anthology/D18-1514.pdf)
2. **Matching Networks for One Shot Learning**. Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra [[link]](https://arxiv.org/abs/1606.04080)
3. **Prototypical Networks for Few-shot Learning**. Jake Snell, Kevin Swersky, Richard SZemel [[link]](https://arxiv.org/abs/1703.05175)
##### Further reading
1. **FewRel 2.0: Towards More Challenging Few-Shot Relation Classification**. Tianyu Gao, Xu Han, Hao Zhu, Zhiyuan Liu, Peng Li, Maosong Sun, Jie Zhou. EMNLP 2019 [[link]](https://www.aclweb.org/anthology/D19-1649.pdf)
2. **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks**. Chelsea Finn, Pieter Abbeel, Sergey Levine [[link]](https://arxiv.org/abs/1703.03400)
3. **Matching the Blanks: Distributional Similarity for Relation Learnin**. Livio Baldini Soares, Nicholas FitzGerald, Jeffrey Ling, Tom Kwiatkowski. ACL 2019 [[link]](https://arxiv.org/pdf/1906.03158.pdf)

## 10. Information Retrieval
#### Must-read papers
1. **PACRR: A Position-Aware Neural IR Model for Relevance Matching**. EMNLP 2017 [[link]](https://arxiv.org/pdf/1704.03940.pdf)
2. **Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval**. ACL 2018 [[link]](https://arxiv.org/abs/1805.07591)
3. **A Deep Look into Neural Ranking Models for Information Retrieval**. 2019 [[link]](https://arxiv.org/pdf/1903.06902.pdf)
4. **Selective Weak Supervision for Neural Information Retrieval**. WWW 2020 [[link]](https://arxiv.org/abs/2001.10382)
#### Further reading
1. **Explicit Semantic Ranking for Academic Search via Knowledge Graph Embedding**. WWW 2017 [[link]](https://dl.acm.org/doi/abs/10.1145/3038912.3052558)
2. **Query suggestion with feedback memory network**. WWW 2018 [[link]](https://dl.acm.org/doi/abs/10.1145/3178876.3186068)
3. **NPRF: A Neural Pseudo Relevance Feedback Framework for Ad-hoc Information Retrieval**. EMNLP 2018 [[link]](https://dl.acm.org/doi/abs/10.1145/3038912.3052558)
4. **Towards Better Text Understanding and Retrieval through Kernel Entity Salience Modeling**. SIGIR 2018 [[link]](https://dl.acm.org/doi/10.1145/3209978.3209982)
5. **Deeper Text Understanding for IR with Contextual Neural Language Modeling**. SIGIR 2019 [[link]](https://dl.acm.org/doi/abs/10.1145/3331184.3331303)

## 11. Question Answering
#### a. Reading Comprehension
1. **SQuAD: 100,000+ Questions for Machine Comprehension of Text**. EMNLP 2016  [[link]](https://www.aclweb.org/anthology/D16-1264.pdf)
2. **Bidirectional Attention Flow for Machine Comprehension**. ICLR 2017  [[link]](https://openreview.net/forum?id=HJ0UKP9ge)
3. **Simple and Effective Multi-Paragraph Reading Comprehension**. ACL 2018  [[link]](https://www.aclweb.org/anthology/P18-1078.pdf)
#### b. Open-domain QA
1. **Reading Wikipedia to Answer Open-Domain Questions**. ACL 2017  [[link]](https://www.aclweb.org/anthology/P17-1171.pdf)
2. **Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text**. EMNLP 2018  [[link]](https://www.aclweb.org/anthology/D18-1455.pdf)
#### c. KBQA
1. **Question Answering with Subgraph Embedding**. EMNLP 2014  [[link]](https://www.aclweb.org/anthology/D14-1067.pdf)
2. **Semantic Parsing via Staged Query Graph Generation: Question Answering with Knowledge Base**. ACL 2015  [[link]](https://www.aclweb.org/anthology/P15-1128.pdf)
#### d. Other Topics
1. **(Multi-hop) Self-Assembling Modular Networks for Interpretable Multi-Hop Reasoning**. EMNLP 2019  [[link]](https://www.aclweb.org/anthology/D19-1455/)
2. **(Symbolic) Neural symbolic Reader: scalable integration of distributed and symbolic representations for reading comprehension**. ICLR 2020  [[link]](https://openreview.net/forum?id=ryxjnREFwH)
3. **(Adversial) Adversarial Examples for Evaluating Reading Comprehension Systems**. EMNLP 2017  [[link]](https://www.aclweb.org/anthology/D17-1215.pdf)
4. **(PIQA) Phrase-indexed question answering: A new challenge for scalable document comprehension**. EMNLP 2018  [[link]](https://www.aclweb.org/anthology/D18-1052.pdf)
5. **(Common Sense) Graph-Based Reasoning over Heterogeneous External Knowledge for Commonsense Question Answering**.  [[link]](https://arxiv.org/abs/1909.05311)
6. **(CQA) SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering**.  [[link]](https://arxiv.org/abs/1812.03593)

## 12. Text Generation
#### a. Survey
1. **Tutorial on variational autoencoders** [[link]](https://arxiv.org/pdf/1606.05908.pdf)
2. **Neural text generation: A practical guide** [[link]](https://arxiv.org/pdf/1711.09534.pdf)
3. **Survey of the state of the art in natural language generation: Core tasks, applications and evaluation** [[link]](https://www.jair.org/index.php/jair/article/download/11173/26378)
4. **Neural Text Generation: Past, Present and Beyond** [[link]](https://arxiv.org/pdf/1803.07133.pdf)
5. **Survey of the state of the art in natural language generation: Core tasks, applications and evaluation** [[link]](https://www.jair.org/index.php/jair/article/view/11173)
#### b. Classic
1. **A neural probabilistic language model** [[link]](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (NNLM)
2. **Recurrent neural network based language model** [[link]](http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf) (RNNLM)
3. **Sequence to sequence learning with neural networks** [[link]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) (seq2seq)
#### c. VAE based
1. **Generating Sentences from a Continuous Space** [[link]](https://arxiv.org/pdf/1511.06349.pdf?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue)
2. **Long and Diverse Text Generation with Planning-based Hierarchical Variational Model** [[link]](https://arxiv.org/pdf/1908.06605v2.pdf)
#### d. GAN based
1. **Adversarial feature matching for text generation** [[link]](https://arxiv.org/pdf/1706.03850.pdf) (TextGAN)
#### e. Knowledge based
1. **Text Generation from Knowledge Graphs with Graph Transformers** [[link]](https://arxiv.org/pdf/1904.02342.pdf)
2. **Neural Text Generation from Rich Semantic Representations** [[link]](https://arxiv.org/pdf/1904.11564)

## 13. Discourse Analysis
#### a. Reference in Language & Coreference Resolution
1. **Unsupervised Models for Coreference Resolution**. Vincent Ng.  EMNLP 2008. [[link]](https://www.aclweb.org/anthology/D08-1067.pdf) 
2. **End-to-end Neural Coreference Resolution**. Kenton Lee, Luheng He, Mike Lewis, Luke Zettlemoyer. EMNLP 2017. [[link]](https://arxiv.org/abs/1707.07045v2) 
3. **Coreference Resolution as Query-based Span Prediction**. Wei Wu, Fei Wang, Arianna Yuan, Fei Wu, Jiwei Li. ACL 2020.  [[link]](https://github.com/ShannonAI/CorefQA)
#### b. Coherence & Discourse Relation Classification
1. **Implicit Discourse Relation Classification via Multi-Task Neural Networks**. Yang Liu, Sujian Li, Xiaodong Zhang, Zhifang Sui. AAAI 2016. [[link]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11831/12018)
2. **Implicit Discourse Relation Detection via a Deep Architecture with Gated Relevance Network**. Jifan Chen, Qi Zhang, Pengfei Liu, Xipeng Qiu, Xuanjing Huang. ACL 2016. [[link]](https://www.aclweb.org/anthology/P16-1163.pdf)
3. **Employing the Correspondence of Relations and Connectives to Identify Implicit Discourse Relations via Label Embeddings**. Linh The Nguyen, Ngo Van Linh, Khoat Than, Thien Huu Nguyen. ACL 2019. [[link]](https://www.aclweb.org/anthology/P19-1411.pdf)
4. **Linguistic Properties Matter for Implicit Discourse Relation Recognition: Combining Semantic Interaction, Topic Continuity and Attribution**. Wenqiang Lei, Yuanxin Xiang, Yuwei Wang, Qian Zhong, Meichun Liu, Min-Yen Kan. AAAI 2018. [[link]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17260/16015)
#### c. Context Modeling and Conversation
1. **A Survey on Dialogue Systems: Recent Advances and New Frontiers**. Hongshen Chen, Xiaorui Liu, Dawei Yin, Jiliang Tang. 2018. [[link]](https://arxiv.org/abs/1711.01731)
2. **A Diversity-Promoting Objective Function for Neural Conversation Models**. Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, Bill Dolan. NAACL 2016. [[link]](https://www.aclweb.org/anthology/N16-1014/)
3. **A Persona-Based Neural Conversation Model**. Jiwei Li, Michel Galley, Chris Brockett, Georgios PSpithourakis, Jianfeng Gao, Bill Dolan. ACL 2016. [[link]](https://arxiv.org/abs/1603.06155)

## 14. Interdiscipline
#### a. Cognitive Linguistics and NLP
1. **Computational Cognitive Linguistics** [[link]](https://www.aclweb.org/anthology/C04-1160.pdf)
2. **Ten Lectures on Cognitive Linguistics by George Lakoff** [[link (access using Tsinghua Laboratory Account)]](https://brill.com/view/title/54941?language=en)
#### b. Psycholinguistics and NLP
1. **A Computational Psycholinguistic Model of Natural Language Processing** [[link]](https://www.aaai.org/Papers/FLAIRS/2004/Flairs04-041.pdf)
2. **Slides of the Cambridge NLP course** [[link]](https://www.cl.cam.ac.uk/teaching/1314/NLP/slides11.pdf)
3. **Reading materials of the MIT course Computational Psycholinguistics** [[link]](http://www.mit.edu/~rplevy/teaching/2017spring/9.19/)
#### c. Sociolinguistics and NLP
1. **Computational Sociolinguistics- A Survey** [[link]](https://www.mitpressjournals.org/doi/pdfplus/10.1162/COLI_a_00258)
2. **Research Topic of Computational Sociolinguistics in Frontiers** [[link]](https://www.frontiersin.org/research-topics/9580/computational-sociolinguistics)
3. **Introduction to Computational Sociolinguistics** [[link]](http://www.rctatman.com/files/Tatman_2019_CompSocio.pdf)







