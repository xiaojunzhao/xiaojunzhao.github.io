---
layout: post
title: Word2Vec Implementation Using Pure Python
permalink: /blog/Word2Vec-Python/
---

<p style='text-align: justify;'>The implementation of Skip-gram model is based on the paper: 
<b>Distributed Representations of Words and Phrases and their Compositionality</b> by Tomas Mikolov, et al from Google.
We'll not only implement skip-gram model but also implement some techniques mentioned in the paper, such as negative sampling, phrase learning,etc, to improve the training speed and accuracy.</p>

<p style='text-align: justify;'> First, let's construct the corpus and vocabulary from data,</p>

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import math
class Corpus(object):
    
    """ Build corpus object for each txt file
    """
    def __init__(self,file_name,num_passes,phrase_threshold,discounting_coeff,phrase_output_filename):
        
        f = open("word2vec/"+file_name,'r')
        i = 0
        tokens = [] 
        for line in f:
            line_tokens = line.split()
            for t in line_tokens:
                t = t.lower()
                if len(t) > 1 and t.isalnum(): # check if a token is alphanumeric and len of token greater than 1
                    tokens.append(t)
            i += 1
            if i % 10000 == 0:
                print("Reading corpus: %d" % i)
            
        f.close()
        self.tokens = tokens
        self.compute_token_cnt_map()
        retained_tokens = []
        for t in self.tokens:
            prob = self.subsampling_frequent_words(t)
            r = random.uniform(0,1)
            if r>prob:
                retained_tokens.append(t)
        self.tokens = retained_tokens
        for x in range(1, num_passes + 1):
            self.phrase_learning(x, discounting_coeff, phrase_threshold, phrase_output_filename)
            self.compute_token_cnt_map()
        self.save_to_file(file_name)
    
    def subsampling_frequent_words(self,t):
        
        prob = 1- math.sqrt(1e-3/(self.unigram_cnt_map[t]/len(self.tokens)))
        return prob
```

<p style='text-align: justify;'> In the above code snippet, we created the token list, and a token-to-count map for each text file. In addition, as paper suggested, we removed the frequent words (such as stop words) by using the subsampling techinque, which discards each word by probability of $$P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}$$.</p>
<p style='text-align: justify;'> We also used `phrase_learning` function to update the token list by forming phrases(words that appear frequently together). It's intuitive that many phrases have a meaning that is not a simple composition of the meanings of its individual words. So it's straightforward for us to treat those phrases as individual tokens. In the paper, the author proposed a data driven approach to learn those phrases,where phrases are formed based on the unigram and bigram counts, using $$ score(w_i,w_j) = \frac{count(w_{i}w_{j})-\delta}{count(w_i) \times count(w_j)}$$ ,where δ is used as a discounting coefficient and prevents too many phrases consisting of very infrequent words to be formed. The bigrams with score above the chosen threshold are then used as phrases. In the code snippet below, we implemented this method for phrase learning: </p>

```python
    def phrase_learning(self,x,discounting_coeff,phrase_threshold,phrase_output_filename):
        
        
        bigrams_cnt_map = self.compute_bigrams()
        phrases = []
        for bigram,cnt in bigrams_cnt_map.items():
            w_i = bigram[0]
            w_j = bigram[1]
            score = (float(cnt)-discounting_coeff)/(self.unigram_cnt_map[w_i]*self.unigram_cnt_map[w_j])
            if score>phrase_threshold:
                phrase = '_'.join(bigram)
                phrases.append(phrase)
                
        # reconstruct token list with the learned phrases
        final_tokens = []
        i = 0
        while i<len(self.tokens):
            if i+1<len(self.tokens):
                w = self.tokens[i]
                next_w = self.tokens[i+1]
                candidate_phrase = w+'_'+next_w
                if candidate_phrase in phrases:
                    final_tokens.append(candidate_phrase)
                    i+=2
                else:
                    final_tokens.append(w)
                    i+=1
            else:
                final_tokens.append(self.tokens[i])
                i+=1
        self.tokens = final_tokens
        
        
    def compute_bigrams(self):
        
        bigrams_cnt_map = {}
        for i in range(len(self.tokens)-1):
            t = (self.tokens[i],self.tokens[i+1])
            if t not in bigrams_cnt_map:
                bigrams_cnt_map[t]=1
            else:
                bigrams_cnt_map[t]+=1
            
            
        return bigrams_cnt_map
        
        
    def save_to_file(self,file_name):
        
        """save preprocessed tokens to a file
        """
        i = 1

        f = open('preprocessed-' + file_name, 'w')
        line = ''
        for token in self.tokens:
            if i % 20 == 0:
                line += token
                f.write('%s\n' % line)
                line = ''
            else:
                line += token + ' '
            i += 1
        
        f.close()
```

<p style='text-align: justify;'>In addition to building the corpus for training, we also need to construct the vocabulary, because we need to know the indices of each word in the Vocabulary. In the code snippet below, we created a attribute `word_map` which maps each word in the vocabulary to its index. </p>

```python
class Vocabulary(object):
    
    def __init__(self,corpus,min_count):
        
        self.words = []
        word_map = {}
        for t in corpus.tokens:
            if t not in word_map:
                word_map[t] = len(self.words)
                self.words.append(t)
        self.word_cnt = Counter(corpus.tokens)
        
        # remove rare words
        self.remove_rare_words(min_count)
        self.word_cnt = Counter(self.words) # update word_cnt
                 
            
    def remove_rare_words(self,min_count):
        
        rare_cnt = 0
        tmp = [('rare',rare_cnt)]
        unk_hash = 0

        count_unk = 0
        for t in self.words:
            if self.word_cnt[t] < min_count:
                count_unk += 1
                rare_cnt += self.word_cnt[t]
                tmp[unk_hash] = ('rare',rare_cnt)
            else:
                tmp.append((t,self.word_cnt[t]))
        tmp.sort(key=lambda x : x[1], reverse=True)

        # Update word_map
        word_map = {}
        for i, (t,_) in enumerate(tmp):
            word_map[t] = i

        self.words = list(zip(*tmp))[0]
        self.word_map = word_map
        
        
    def __len__(self):
        return len(self.words)
```

<p style='text-align: justify;'>In this paper, the author also proposed <b>negative sampling</b> approach to make the training process faster, because in the original softmax function, one needs to sum over the product of word vector of each word in the vocabulary and the input word vector, which is really computationally expensive. See below:</p>
<div class='row'>
    <img src = "/images/word2vecsoftmax.png" width="300">
</div>
<p style='text-align: justify;'>Instead of computing gigantic summation over the entire vocabulary, the author proposed the negative sampling cost function, which only updates the gradients for center word, outside words and negative samples. $$E = -log\sigma({V_{wo}^{\prime}}^{T}V_{wi})-\sum_{w_j\in w_{neg}} log\sigma(-{V_{wj}^{\prime}}^{T}V_{wi})$$</p>
<p style='text-align: justify;'>The idea behind negative sampling  is to maximize the similarity between center word and outside words, and minimize the similarity between center word and randomly selected negative words. And the selection of the random words is based on the probability: $$\frac{U(w)^{3/4}}{Z}$$,where U(w) is the unigram distribution, and Z is the normalization constant. The scripts for negative sampling is below: </p>

``` python
class NegativeSamplingTable(object):
    
    def __init__(self,vocab):
        power = 0.75
        # normalization constant Z
        Z = sum([math.pow(vocab.word_cnt[t],power) for t in vocab.words])
        table_size = int(1e8)
        table = np.zeros(table_size)
        p = 0
        i = 0
        for j,t in enumerate(vocab.words):
            p+=float(math.pow(vocab.word_cnt[t],power))/Z
            while i<table_size and float(i)/table_size<p:
                table[i] = j
                i+=1
        self.table = table
        
    def sample(self,count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]    
```

<p style='text-align: justify;'>To better understanding the computation of gradient updates for nn weights, let's first look at how the skip-gram model looks like: </p>
<img src = "/images/skipgram.png" width="800">
<p style='text-align: justify;'>We need to derive gradients for both \(W^{\prime}\) and \(W\). The specific details of gradient derivation for negative sampling cost function is shown below: $$\frac{\partial E}{\partial V_{wj}^{\prime}} = \frac{\partial E}{\partial {V_{wj}^{\prime}}^{T}V_{wi}}.\frac{\partial {V_{wj}^{\prime}}^{T}V_{wi}}{\partial V_{wj}^{\prime}}=\big(\sigma({V_{wj}^{\prime}}^{T}V_wi)-t_j\big)V_{wi} $$, where \(t_{j}\) is the “label” of word \(w_{j}\) . \(t\) = 1 when \(w_j\) is a positive sample; \(t\) = 0 otherwise, which results in the following updates equation of the output vector: $${V_{wj}^{\prime}}^{new} ={V_{wj}^{\prime}}^{old}-\alpha \big(\sigma({V_{wj}^{\prime}}^{T}V_wi)-t_j\big)V_{wi} $$. To backpropagate the error to the hidden layer and thus update the input vectors of words, we need to take the derivative of E with regard to the hidden layer’s output, obtaining $$\frac{\partial E}{\partial W_{vi}} = \sum_{w_j\in w_o \cup W_{neg}} \big(\sigma({V_{wj}^{\prime}}^{T}V_{wi}-t_{j})V_{wj}^{\prime}\big) $$. In the following code, we're gonna put everything together to train the one layer skip-gram model:</p>

```python
def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))
```

```python

if __name__=='__main__':

    file_name = 'input.txt' 
    k_neg_samples = 5
    min_count = 1 # minimum count for words to be used in the model
    num_passes = 3
    discounting_coeff = 3
    phrase_threshold = 1e-4
    corpus = Corpus(file_name, num_passes, phrase_threshold,discounting_coeff, 'phrases-%s' % file_name)
    vocab = Vocabulary(corpus,min_count)
    table = NegativeSamplingTable(vocab)
    
    # parameters initializaation
    win_size = 5
    emb_dim = 100
    # nn initialization
    W_input = np.random.uniform(low=-0.5/emb_dim, high=0.5/emb_dim, size=(len(vocab), emb_dim))
    W_output = np.zeros(shape=(len(vocab), emb_dim))
    init_alpha = 0.01 # initial learning rate, later the learning rate will decay during the training process
    alpha = init_alpha

    tokens = corpus.tokens
    token_ids = [ vocab.word_map[t] if t in vocab.words else 0 for t in tokens ] 
    
    word_count = 0
    for i,token in enumerate(token_ids):

        context_start = max(i - win_size, 0)
        context_end = min(i + win_size + 1, len(tokens))
        context_words = token_ids[context_start:i] + token_ids[i+1:context_end]   
        if word_count % 10000 == 0:
                print(word_count)
        for context_word in context_words:


            z = np.dot(W_input[token], W_output[context_word])
            p = sigmoid(z)
            g = alpha * (p-1)
            W_output[context_word] -=g*W_input[token] # update weights for the context word 
            neg_samples = [int(target) for target in table.sample(k_neg_samples)]

            #update weights for negative samples
            for neg_sample in neg_samples:
                z = np.dot(W_input[token],W_output[neg_sample])
                p = sigmoid(z)
                g = alpha*p
                W_output[neg_sample] -=g*W_input[token]
                
            # update weights for center word, which is token in our case
            g = 0   
            for target in [context_word]+neg_samples:
                z = np.dot(W_input[token],W_output[target])
                p = sigmoid(z)
                if target==context_word:
                    g += alpha*(p-1)
                else:
                    g += alpha*p
                W_input[token] -=g*W_output[target]


        word_count+=1
```

To visualize the word vector in 2 dimensional space, we use PCA to reduce the dimensionality of computed word vectors.

```python
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X = np.array([get_word_vec(W_input,W_output,t) for t in list(vocab.word_map.keys())[:5000]])
    pca.fit(X)
    
    
    words =['australia','scotland','vancouver','china','russia','poland','italy','france','beijing',\
            'hong_kong','paris','monday','tuesday','friday','saturday','wednesday','thursday']

    coords = np.array([pca.transform(get_word_vec(W_input,W_output,w).reshape(1,-1))[0] for w in words])
    df = pd.DataFrame({
    'x': coords[:,0],
    'y': coords[:,1],
    'word': words
    })
    sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="+", color="skyblue")
    p1=sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':100})

    # add annotations one by one with a loop
    for line in range(0,df.shape[0]):
        p1.text(df.x[line]-0.001, df.y[line], df.word[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
    plt.show()
    
```
<p style='text-align: justify;'> The visualization is shown below, we can see that week days are clustered together and places are clustered together. </p>
<img src = "/images/word2vecviz.png" width="800">

### References: 
1. Rong, Xin. "word2vec parameter learning explained." arXiv preprint arXiv:1411.2738 (2014).
2. Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.
3. https://github.com/tscheepers/word2vec 

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script>
