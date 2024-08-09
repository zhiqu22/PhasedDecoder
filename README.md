# PhasedDecoder

This is the repository for the paper:  
**Improving Language Transfer Capability of Decoder-only Architecture in Multilingual Neural Machine Translation**

---

Prerequisite:
1. install [Fairseq](https://github.com/facebookresearch/fairseq)
2. download [Moses](https://github.com/moses-smt/mosesdecoder)
3. build TED-19 and OPUS-100 dataset as instructions [here](https://github.com/zhiqu22/ZeroTrans)
4. run commands
```bash
mkdir fairseq/models
mv PhasedDecoder fairseq/models/PhasedDecoder
mkdir moses
mv mosesdecoder/scripts moses
```
These files are ordered as following:
```
├── fairseq/  
│     └── models/  
├── pdec_work/  
│     ├── train/  
│     ├── ted_evaluation/  
│     ├── opus_evaluation/  
│     ├── logs(automatically make)/  
│     ├── results(automatically make)/  
│     └── checkpoints(automatically make)/  
├── moses/  
├── TED-BIN/  
└── OPUS-BIN/  
```
Finally, you can run the command to reproduce the results.
```bash
cd pdec_work
bash train/ted.sh
bash train/opus.sh
# note:
# please edit the variables "ROOT_PATH" and "DATA_BIN" in those two files as you need.
```
