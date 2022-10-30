# ME
microexpression recognition.<br>
This repo is to realize short time mirco expression recognition in realtime. Base model is LSTM and dataset for pretrain is CK+.<br>
Pretrain is 10 fold train.<br>

### LSTM attention
hn of lstm can be regarded as video level feature.<br>
self attention weight is α = σ(linear(lms)).<br>
relation attention weight is β = σ(linear([lms:hn])).<br>
so, a sample aggregated feature is Σαβ[lms:hn]/Σαβ.<br>

### Transformer