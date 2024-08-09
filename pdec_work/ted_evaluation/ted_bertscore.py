from bert_score import BERTScorer
import sys
import os

method_name = sys.argv[1]
model_id = sys.argv[2]
tgt = sys.argv[3]
cuda_id = sys.argv[4]
save_path = sys.argv[5]
save_path = os.path.join(save_path, "results")

language_sequence = ["en", "ar", "he", "ru", "ko", "it", "ja", "zh", "es", "nl", "vi", "tr", "fr", "pl", "ro", "fa", "hr", "cs", "de"]


def _read_txt_strip_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]

tmp_score = BERTScorer(lang=tgt, device="cuda:{}".format(cuda_id))

writing_list = []
for src in language_sequence:
    if src == tgt: continue
    ref = _read_txt_strip_(os.path.join(save_path, str(method_name), str(model_id), "{}-{}.detok.r".format(src, tgt)))
    hypo = _read_txt_strip_(os.path.join(save_path, str(method_name), str(model_id), "{}-{}.detok.h".format(src, tgt)))
    P, R, F = tmp_score.score(hypo, ref, batch_size=100)
    P, R, F = round(P.mean().item() * 100, 2), round(R.mean().item() * 100, 2), round(F.mean().item() * 100, 2)
    writing_list.append("{}-{}\n".format(src, tgt))
    writing_list.append("P: {} R: {} F: {} \n".format(P, R, F))


file = open(os.path.join(save_path, str(method_name), str(model_id), "{}.bertscore".format(str(model_id))), 'a', encoding='utf-8')
file.writelines(writing_list)
file.close()




