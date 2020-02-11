## Model backlog (list of the developed model and it's score)
- **Train** and **validation** are the splits using the train data from the competition.
- The competition metric is **Mean Columnwise Spearman's r (rank correlation coefficient)**.
- **Runtime** is the time in seconds that the kernel took to finish.
- **Pb Leaderboard** is the Public Leaderboard score.
- **Pv Leaderboard** is the Private Leaderboard score.

---

## Models

|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|
|-----|-----|----------|--------------|--------------|
|1 - GoogleQ&A - USE-QA|0.491|0.391|0.352|???|
|2 - GoogleQ&A - USE-QA lr3e-3|0.507|0.390|0.345|???|
|3 - GoogleQ&A - USE|0.480|0.379|0.334|???|
|4 - GoogleQ&A - USE-Large|0.501|0.393|0.354|???|
|5 - GoogleQ&A - USE-Large BS64|???|???|???|???|
|6 - GoogleQ&A - USE-Large BS16|0.494|0.392|0.353|???|
|7 - GoogleQ&A - USE-Large Elu|0.465|0.382|0.341|???|
|8 - GoogleQ&A - USE-Large Swish|0.464|0.384|0.344|???|
|9 - GoogleQ&A - USE-Large ML split|0.484|0.384|0.355|???|
|10-GoogleQ&A Train-USE|0.495|0.372|0.335|???|
|11-GoogleQ&A Train-USE|0.467|0.317|0.298|???|
|12-GoogleQ&A Train-USE trim|0.493|0.373|0.335|???|
|13-GoogleQ&A Train-USE lower|0.493|0.373|000|???|
|14-GoogleQ&A Train-USE ?!|0.493|0.373|0.335|???|
|15-GoogleQ&A Train-USE selu|0.451|0.360|0.318|???|
|16-GoogleQ&A Train-USE digits|0.493|0.373|0.336|???|
|17-GoogleQ&A Train-USE punctuation|0.493|0.368|0.327|???|
|18-GoogleQ&A Train-USE alphanumeric|0.495|0.374|0.332|???|
|19-GoogleQ&A Train-USE misspelling|0.492|0.373|0.334|???|
|20-GoogleQ&A Train-USE contraction|0.493|0.373|0.334|???|
|21-GoogleQ&A Train-USE stopwords|0.482|0.342|0.298|???|
|22-GoogleQ&A Train-USE text pre-process|0.485|0.371|0.331|???|
|23-GoogleQ&A Train-USE text pre-process2|0.492|0.373|0.332|???|
|24-GoogleQ&A Train-USE-QA text pre-process2|0.487|0.380|0.355|???|
|25-GoogleQ&A Train-USE 2Dense|0.479|0.354|0.311|???|
|26-GoogleQ&A Train-USE 2Dense2|0.508|0.372|0.323|???|
|27-GoogleQ&A Train-USE Embedding Dense|0.491|0.375|0.324|???|
|28-GoogleQ&A Train-USE Embedding Dropout|0.490|0.370|0.323|???|
|29-GoogleQ&A Train-USE Embedding DropoutDense|0.482|0.378|0.327|???|
|30-GoogleQ&A Train-USE Embedding SubModules|0.462|0.366|0.317|???|
|31-GoogleQ&A Train-USE Multi-Sample Dropout|0.442|0.365|0.319|???|
|32-GoogleQ&A Train-USE Multi-Sample Dropout2|0.463|0.365|0.319|???|
|33-GoogleQ&A Train-USE|0.279|0.265|0.242|???|
|34-GoogleQ&A Train-USE early stopping|0.478|0.370|0.331|???|
|35-GoogleQ&A Train-USE category|0.479|0.372|0.331|???|
|36-GoogleQ&A Train-USE category|0.472|0.368|0.333|???|
|37-GoogleQ&A Train-USE lecun_normal|0.478|0.369|0.336|???|
|38-GoogleQ&A Train-USE he_normal|0.462|0.365|0.334|???|
|39-GoogleQ&A Train-USE glorot_uniform|0.474|0.368|0.334|???|
|40-GoogleQ&A Train-USE glorot_uniform2|0.490|0.372|0.335|???|
|41-GoogleQ&A Train-USE lecun_normal2|0.492|0.374|0.331|???|
|42-GoogleQ&A Train-USE text features|0.450|0.360|0.322|???|
|43-GoogleQ&A Train-USE text features scaled|0.512|0.373|0.325|???|
|44-GoogleQ&A Train-USE misspelings count|0.489|0.371|0.331|???|
|45-GoogleQ&A Train-USE unique words|0.490|0.371|0.331|???|
|46-GoogleQ&A Train-USE TF-IDF bigram|0.558|0.363|0.311|???|
|47-GoogleQ&A Train-USE TF-IDF trigram|0.558|0.361|0.311|???|
|48-GoogleQ&A Train-USE skip|0.561|0.361|0.317|???|
|49-GoogleQ&A Train-USE-QA skip|0.555|0.368|0.335|???|
|50-GoogleQ&A Train-USE skip concat|0.470|0.363|0.321|???|
|51-GoogleQ&A Train-USE skip add|0.447|0.347|0.308|???|
|52-GoogleQ&A Train-USE Add|0.420|0.351|0.326|???|
|53-GoogleQ&A Train-USE GlobalAvg|0.476|0.363|0.320|???|
|54-GoogleQ&A Train-USE LSTM|0.460|0.365|0.321|???|
|55-GoogleQ&A Train-USE GRU|0.460|0.361|0.321|???|
|56-GoogleQ&A Train-USE CNN|0.476|0.366|0.324|???|
|57-GoogleQ&A Train-USE LSTM|0.479|0.373|0.330|???|
|58-GoogleQ&A Train-USE LSTM+GlobalAvg|0.480|0.373|0.330|???|
|59-GoogleQ&A Train-USE 2xLSTM|0.470|0.371|0.329|???|
|60-GoogleQ&A Train-USE BiLSTM|0.472|0.370|0.330|???|
|61-GoogleQ&A Train-USE+LSTM|0.517|0.314|0.264|???|
|62-GoogleQ&A Train-USE+LSTM v2|0.52|0.318|0.268|???|
|63-GoogleQ&A Train-USE+nnlm128|0.47|0.367|0.328|???|
|64-GoogleQ&A Train-USE+WikiWords500|0.436|0.358|0.323|???|
|65-GoogleQ&A Train-USE-QA+nnlm128|0.48|0.38|0.347|???|
|66-GoogleQ&A Train-USE+Glove GlobalAVG|0.476|0.377|0.332|???|
|67-GoogleQ&A Train-USE+Glove GlobalAVG Uncased|0.477|0.378|0.332|???|
|68-GoogleQ&A Train-USE+Glove LSTM Uncased|0.483|0.353|0.321|???|
|69-GoogleQ&A Train-USE+Glove LSTM Uncased|0.503|0.38|0.331|???|
|70-GoogleQ&A Train-USE+Glove LSTM-dense uncased|0.476|0.37|???|???|
|71-GoogleQ&A Train-Glove GlobalAVG Uncased|0.379|0.333|0.292|???|
|72-GoogleQ&A Train-Glove LSTM Uncased|0.418|0.349|0.304|???|
|73-GoogleQ&A Train-Glove 2LSTM Uncased|0.426|0.317|0.270|???|
|74-GoogleQ&A Train-Glove 2LSTM MAX_FEAT=20000|0.433|0.318|0.277|???|
|75-GoogleQ&A Train-Glove 2LSTM MAX_LEN=200|0.429|0.32|0.275|???|
|76-GoogleQ&A Train-Glove 2LSTM AVG|0.404|0.341|0.302|???|
|77-GoogleQ&A Train-Glove 2LSTM Dense|0.357|0.3|0.274|???|
|78-GoogleQ&A Train-Glove 2LSTM AVGv2|0.407|0.341|0.296|???|
|79-GoogleQ&A Train-Glove BiLSTM|0.428|0.332|0.292|???|
|80-GoogleQ&A Train-Glove LSTM MAX_FEAT 3k|0.4|0.342|0.302|???|
|81-GoogleQ&A Train-Glove LSTM MAX_FEAT 100k|0.42|0.351|0.308|???|
|82-GoogleQ&A Train-Glove LSTM MAX_LEN 300|0.412|0.35|0.311|???|
|83-GoogleQ&A Train-Glove LSTM 10paragraphs|0.449|0.334|0.298|???|
|84-GoogleQ&A Train-Glove LSTM 10paragraphs v2|0.469|0.324|0.280|???|
|85-GoogleQ&A Train-Glove LSTM|0.457|0.348|0.308|???|
|86-GoogleQ&A Train-Glove Category&host embedding|0.434|0.34|0.287|???|
|87-GoogleQ&A Train-Glove text features|0.416|0.348|0.287|???|
|88-GoogleQ&A Train-Glove meta features|0.44|0.349|0.308|???|
|89-GoogleQ&A Train-Glove RAdam|0.403|0.334|0.289|???|
|90-GoogleQ&A Train-Glove SGD One Cycle|0.365|0.328|0.284|???|
|91-GoogleQ&A Train-Glove SGD Cosine|0.367|0.332|0.294|???|
|92-GoogleQ&A Train-Glove Adam Cosine|0.443|0.328|0.278|???|
|93-GoogleQ&A Train-Glove SGDR|0.364|0.329|0.287|???|
|94-GoogleQ&A Train-Glove SGD Cyclical_triangular2|0.347|0.318|0.277|???|
|95-GoogleQ&A Train-Bert-base clean|0.651|0.389|0.354|???|
|96-GoogleQ&A Train-Bert-base contractions|0.65|0.385|0.357|???|
|97-GoogleQ&A Train-Bert-base misspellings|0.648|0.384|0.352|???|
|98-GoogleQ&A Train-Bert-base|0.508|0.387|0.339|???|
|99-GoogleQ&A Train-Bert-base RAdam|0.461|0.373|0.327|???|
|100-GoogleQ&A Train-Bert-base RAdam2|0.468|0.376|0.329|???|
|101-GoogleQ&A Train-Bert-base RAdam3|0.468|0.377|0.329|???|
|102-GoogleQ&A Train-Bert-base Adam Cosine|0.459|0.372|0.323|???|
|103-GoogleQ&A Train-Bert-base clean|0.508|0.387|0.341|???|
|104-GoogleQ&A Train-Bert-base raw|0.51|0.39|0.344|???|
|105-GoogleQ&A Train-Bert-base pooled_output|0.496|0.378|0.355|???|
|106-GoogleQ&A Train-Bert-base spatial_dropout|0.512|0.392|0.345|???|
|107-GoogleQ&A Train-USE-QA text pre-process2 3Fold|0.498|0.376|0.356|???|
|108-GoogleQ&A Train-USE-QA text pre-process2 5Fold|000|000|0.356|???|
|109-GoogleQ&A Train-Bert-base no dropout|0.504|0.38|0.350|???|
|110-GoogleQ&A Train-Bert_base_uncased no_title|0.488|0.359|0.317|???|
|111-GoogleQ&A Train-Bert-base no dropout v2|0.505|0.381|0.351|???|
|112-GoogleQ&A Train-Bert_base_uncased seq slices|0.502|0.379|0.338|???|
|113-GoogleQ&A Train-Bert_base_uncased|0.452|0.375|0.331|???|
|114-GoogleQ&A Train-Bert_base_uncased seq slicesV2|0.498|0.391|0.355|???|
|115-GoogleQ&A Train-Bert_base_uncased Quest/Ans|0.496|0.391|0.347|???|
|116-GoogleQ&A Train-Bert_base_uncased-3F LRWarmup1|0.506|0.388|0.375|0.353|
|117-GoogleQ&A Train-Bert_base_uncased-3F LRWarmup2|0.514|0.391|0.375|0.354|
|118-GoogleQ&A Inf-3Fold-Bert_base_unc Raw|000|000|000|000|
|119-GoogleQ&A Train-3Fold-Bert_base_unc category|0.514|0.381|0.365|0.344|
|120-GoogleQ&A Train-3Fold-Bert_base_unc host|0.545|0.382|0.362|0.344|
|121-GoogleQ&A Train-3Fold-Bert_base_unc categoryV2|0.373|0.343|0.328|0.307|
|122-GoogleQ&A Train-3Fold-Bert_base_unc categoryV3|0.433|0.373|0.352|0.332|
|123-GoogleQ&A Train-3Fold-Bert_base_unc categoryV4|0.460|0.379|0.356|0.337|
|124-GoogleQ&A Train-3Fold-Bert_base_unc head/tail|0.509|0.389|0.373|0.352|
|125-GoogleQ&A Train-2Fold-Bert_base_unc categoryV5|0.465|0.381|0.353|0.329|
|126-GoogleQ&A Train-1Fold-Bert_base_unc Quest/Ans|0.503|0.429|0.377|0.350|
|127-GoogleQ&A Train-1Fold-Bert_base_unc Quest/Ans2|0.499|0.402|0.371|0.346|
|128-GoogleQ&A Train-1Fold-Bert_base_unc Quest/Ans3|0.486|0.401|0.367|0.344|
|129-GoogleQ&A Train-3Fold-Bert_base_unc Question Answer|000|000|0.379|0.359|
|130-GoogleQ&A Train-3Fold-Bert_base_unc Quest/Ans3|0.487|0.397|0.364|0.383|
