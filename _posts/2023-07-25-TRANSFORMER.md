---
title: Transformer Tutorial
date: 2023-07-25
categories: [Machine Learning]
tags: [transformer,llms]     # TAG names should always be lowercase
math: true
mermaid: true
---
Transformers have been proposed in the paper "Attention is all you need" by Ashish Vaswani *et al.* {% cite attn --file ref_transformer %}. Transformers are intrinsically related to Large Language Models (LLMs), as LLMs are built upon the transformer architecture.

<p style="text-align:center;"><em>But what exactly are transformers?</em></p> 

Essentially, transformers are neural networks designed for sequential data or time series. What distinguishes them from other neural networks designed for the same purposes are the facts that transformers are non recurrent models and use the attention mechanism.

Because of these two main characteristics they offer several advantages compared to LSTM:
- Handle long-range dependencies thanks to the self-attention mechanism
- Faster than LSTM because they can process the input sequence in parallel
- Easier to train thanks to the absence of recurrent connections, they do not suffer from exploding or vanishing gradient issues
- Generate more meaningful representations that better capture the relationships between words and their contexts compared to the LSTM's fixed-size hidden state (in natural language processing) 

In this tutorial, I'm going to walk you through understanding the transformer model. Given its complexity, I'll start with the basics and gradually dive into more intricate details of the model step by step.

## The Model
<div style="display:none">
$
\newcommand{\vect}[1]{\boldsymbol{#1}}
\newcommand{\vW}{\vect{W}}
\newcommand{\vM}{\vect{M}}
\newcommand{\vx}{\vect{x}}
\newcommand{\vz}{\vect{z}}
\newcommand{\coloneqqf}{\mathrel{\vcenter{:}}=}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
$
</div>
When having a first glimpse of the model, we see that it consists of a stack of encoders and decoders. Additionally, we see that each encoder contain a self-attention block, whereas the decoder has both a *(masked)* self-attention block and a so called encoder-decoder attention block.

![Figure 1](/assets/transformer/first_glimpse.png){:width="100%"}
_Figure 1: First glimpse of the model._


By having a closer look at the transformer encoder in [Figure 2](/assets/transformer/encoder.png) we notice that the words (represented as embeddings) flow in parallel.
Additionally, we can see that the embeddings are handled independently except for the self-attention block, which creates dependencies between the different embeddings. Also, **so far**, the position of the different words does not matter.

![Figure 2](/assets/transformer/encoder.png){:width="80%"}
_Figure 2: Transformer encoder._


### Self-attention
For each embedding we compute query, key and value, by multiplying with the trainable parameters $\vW_q$, $\vW_k$, and $\vW_v$, respectively. As illustrated in [Figure 3](/assets/transformer/qkv.png) we can see that the parameters are shared between the different embeddings.

![Figure 3](/assets/transformer/qkv.png){:width="50%"}
_Figure 3: Computing queries, keys, and values._

Afterwards, we proceed to compute self-attention. As depicted in [Figure 4](/assets/transformer/self_attn.png), for each embedding, we calculate attention scores by multiplying the corresponding query with the keys. These scores are then transformed into attention weights through a softmax operation.

Then the new embeddings are abtained as a weighted sum of the values.
![Figure 4](/assets/transformer/self_attn.png){:width="70%"}
_Figure 4: Self-attention._

<p style="text-align:center;"><em>But how does attention mechanism enhance the embeddings?</em></p> 

Attention enhances embeddings in neural networks by introducing a context-aware weighting system that leverages other embeddings to improve the encoding of the current embedding. For instance, upon examining the changes from $\vx_1$ to $\vz_1$, it becomes evident that $\vz_1$ now integrates additional information derived from the other embeddings.

In essence, attention allows the transformer to dynamically emphasize relevant embeddings while downplaying less relevant ones, thereby refining the representation of each embedding based on the context provided by others. This capability significantly boosts the network's ability to capture intricate relationships and dependencies within data, making it particularly powerful in tasks requiring nuanced understanding and context-aware processing.

In this example, we can see that the embedding corresponding to the pronoun "it" focuses most on the embedding corresponding to the word "animal".

![Figure 5](/assets/transformer/example.png){:width="30%"}
_Figure 5: Self-attention example._

### Positional encoding
The model presented so far is permutation invariant, in the sense that a permutation of the input embeddings (shuffling of the words of the input sentence) corresponds to an equivalent permutation of the embeddings at the encoder outputs, and does not change the content of each embedding.

Therefore, we need a way to account for the order of the words in the input sequence.
In LSTM this is done automatically because of the recurrent/sequential nature of the model itself (inputs are fed one by one following their order of appearance).

To account for the order of the words we add a vector with a specific pattern to each input embedding, see [Figure 6](/assets/transformer/pos_enc.png). 

![Figure 6](/assets/transformer/pos_enc.png){:width="50%"}
_Figure 6: Positional encoding overview._

The positional encoding for each embedding is obtained using the formula below:

![formula](/assets/transformer/formula.png){:width="40%"}

where the index $i$ varies from $0$ up to $\frac{d_{\text{model}}}{2} - 1$. In particular, we can observe that when $i \ge 1$ the denominator is larger than $1$ and the frequency of the sinusoids decreases.

Let's visualize this with an example. In this example we assume that $d_{\text{model}} = 50$ and we observe what happens to the values of $pos \in \\{0, 1, 2, 3\\}$.

![Figure 7](/assets/transformer/pos-enc-example.png){:width="80%"}
_Figure 7: Positional encoding example._

If we then construct a matrix that has the $pos$ index on the y-axis and the $i$ index on the x-axis we can observe the pattern in [Figure 8](/assets/transformer/pos-enc-pattern.png).

![Figure 8](/assets/transformer/pos-enc-pattern.png){:width="70%"}
_Figure 8: Positional encoding pattern._

In particular, we can observe that the values of the first 20 positions for higher indices of the embedding dimension remain nearly constant.

<p style="text-align:center;"><em>What this mean?</em></p> 

To have an intution we can think about binary numbers. In [Figure 8](/assets/transformer/binary.png), we can see that numbers closer to each other share more of the most significant digits. Similarly, with positional encoding, position indices that are close to each other share the values of most of the indices of the embedding dimension.

![Figure 8](/assets/transformer/binary.png){:width="70%"}
_Figure 8: Positional encoding intuition._

Another motivation for positional encoding is given in the paper: *“We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.”*

In essence, what the authors are saying is that we can always find a $2 \times 2$ matrix $\vM$ that only depends on $k$ such that

<div style="overflow-x: auto; overflow-y: hidden; white-space: nowrap;">
$$\begin{equation*}
\vM \begin{bmatrix} \sin(\omega_i \cdot pos) \\ \cos(\omega_i \cdot pos) \end{bmatrix} = \begin{bmatrix} \sin(\omega_i \cdot (pos+k)) \\ \cos(\omega_i \cdot (pos+k)) \end{bmatrix}
\end{equation*}
$$
</div>

where $\omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}$.

### Transformer decoder

The transformer decoder uses both a masked self-attention and an encoder-decoder attention.

<p style="text-align:center;"><em>Why masking?</em></p> 

In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This means we can't predict words based on future words. 
Masking future positions (setting them to `-inf`) before the softmax step in the self-attention calculation ensures this restriction is enforced.

In Pytorch we would simply do 

```python
    attn_weights = F.softmax(scores + mask, dim=-1)
```

This procedure is very important because it allows us to train the decoder in parallel. However, during the operational time, the inference in the decoder is done sequentially until the embedding of “end of sentence” is reached.

The *Encoder-Decoder Attention* layer, instead, works like the self-attention, except it creates its Queries matrix from the layer below it,  and takes the Keys and Values matrix from the output of the encoder.


### Transformer in action

[^1]: The first decoder input is the "start of sentence" embedding (in the animation is not shown)

In [Figure 9](/assets/transformer/transformer_decoding.gif) we see how the transformer predicts the sentence word by word[^1]. In the final layer of the decoder the softmax is used to take the word with the highest probability.


![Figure 9](/assets/transformer/transformer_decoding.gif){:width="70%"}
_Figure 9: Transformer in action._

<p style="text-align:center; color:#FF69B4;"><em>Is this the whole story? :thinking: </em></p> 


Indeed the model has some more details...


## More details

[^2]: Also the decoder uses positional encoding, residual connections and LayerNorm.


If we have a closer look at the encoder block[^2] we see that it contains also residual connections and layer normalization.

![Figure 10](/assets/transformer/res-conn.png){:width="60%"}
_Figure 10: Additional components._

The main purpose of using residual connections in transformers is to address the vanishing gradient problem. During the backpropagation process, gradients can become extremely small as they are propagated backward through many layers. As a result, the weights in the early layers of the network may not get updated effectively, leading to slow or stalled learning and difficulty in training very deep networks.
The residual connections also preserve the positional encoding information which is added only at the beginning of the model


<p style="text-align:center;"><em>Why LayerNorm instead of BatchNorm?</em></p> 


![Figure 11](/assets/transformer/layernorm.png){:width="40%"}
_Figure 11: LayerNorm vs. BatchNorm._

In [Figure 11](/assets/transformer/layernorm.png) we see the differences between LayerNorm and BatchNorm. In essence LayerNorm is preferred for the following reasons:
- Input sequences can have varying lengths. BatchNorm, which relies on batch statistics, is not suitable
- Order of the words in a sentence matters
- During decoding, we process one embedding at a time
- More time-efficient


### Multi-head attention

Once we understand the concept of self-attention, the idea of multi-head attention becomes *almost* straightforward.

The motivation for multi-head attention is explained in the paper: *"Instead of performing a single attention, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections."*

Furthermore, the paper states: *On each of these projected versions of queries, keys and values we then perform the attention function in parallel. The output values are concatenated and once again projected, resulting in the final values.*
This procedure is illustrated in [Figure 12](/assets/transformer/mha.png).

![Figure 12](/assets/transformer/mha.png){:width="60%"}
_Figure 12: Multi-head attention._

And that wraps up my post!

<p style="text-align:center;"><em>Thank you for your attention! :hugs: </em></p> 

If you enjoyed it, spread the love with a :heart: or share your thoughts below!


{% bibliography --cited --file ref_transformer %}

The figures, onto which I've added my notes, have been extracted from:
- [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- [https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers](https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers)
- [https://kazemnejad.com/blog/transformer_architecture_positional_encoding/](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
