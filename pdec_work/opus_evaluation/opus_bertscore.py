from bert_score import BERTScorer
import sys
import os

method_name = sys.argv[1]
model_id = sys.argv[2]
# 2en, en2, zero
flag = sys.argv[3]
cuda_id = sys.argv[4]
work_path = sys.argv[5]
result_path = os.path.join(work_path, "results")

# en is excluded
language_sequence = ["af", "ar", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy", "da",
                    "de", "el", "es", "et", "eu", "fi", "fr", "ga", "gl", "gu", "he", "nn",
                    "hi", "hr", "hu", "id", "is", "it", "ja", "ka", "kk", "kn", "ko", "lt",
                    "lv", "mg", "mk", "ml", "mr", "ms", "my", "ne", "nl", "oc", "pa", "pl",
                    "pt", "ro","ru", "sh", "sk", "sl", "sq", "sr", "sv", "ta", "te", "tg", 
                    "tr", "tt", "uk", "ur", "uz", "vi", "zh", "no", "nb", ]
zero_pairs = [["de", "nl"], ["nl", "zh"], ["ar", "nl"],
              ["ru", "zh"], ["fr", "nl"], ["de", "fr"],
              ["fr", "zh"], ["ar", "ru"], ["ar", "zh"],
              ["ar", "fr"], ["de", "zh"], ["fr", "ru"],
              ["de", "ru"], ["nl", "ru"], ["ar", "de"]]

def _read_txt_strip_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]

def _compute_bertscore_(tmp_score, result_path, method_name, model_id, src, tgt):
    ref = _read_txt_strip_(os.path.join(result_path, str(method_name), str(model_id), f"{src}-{tgt}.r"))
    hypo = _read_txt_strip_(os.path.join(result_path, str(method_name), str(model_id), f"{src}-{tgt}.h"))
    P, R, F = tmp_score.score(hypo, ref, batch_size=100)
    P, R, F = round(P.mean().item() * 100, 2), round(R.mean().item() * 100, 2), round(F.mean().item() * 100, 2)
    return P, R, F

writing_list = []
if flag == "2en":
    # XLMR
    tmp_score = BERTScorer(lang="en", device=f"cuda:{cuda_id}")
    tgt = "en"
    for src in language_sequence:
        P, R, F =  _compute_bertscore_(tmp_score, result_path, method_name, model_id, src, tgt)
        writing_list.append("{}-{}\n".format(src, tgt))
        writing_list.append("P: {} R: {} F: {} \n".format(P, R, F))
else:
    # bert-base-multilingual-cased
    tmp_score = BERTScorer(lang="others", device=f"cuda:{cuda_id}")
    if flag == "en2":
        src = "en"
        for tgt in language_sequence:
            P, R, F =  _compute_bertscore_(tmp_score, result_path, method_name, model_id, src, tgt)
            writing_list.append("{}-{}\n".format(src, tgt))
            writing_list.append("P: {} R: {} F: {} \n".format(P, R, F))
    else:
        for pair in zero_pairs:
            src, tgt = pair[0], pair[1]
            P, R, F =  _compute_bertscore_(tmp_score, result_path, method_name, model_id, src, tgt)
            writing_list.append("{}-{}\n".format(src, tgt))
            writing_list.append("P: {} R: {} F: {} \n".format(P, R, F))
            tgt, src = pair[0], pair[1]
            P, R, F =  _compute_bertscore_(tmp_score, result_path, method_name, model_id, src, tgt)
            writing_list.append("{}-{}\n".format(src, tgt))
            writing_list.append("P: {} R: {} F: {} \n".format(P, R, F))
            print(writing_list)


file = open(os.path.join(result_path, str(method_name), str(model_id), "{}.bertscore".format(str(model_id))), 'a', encoding='utf-8')
file.writelines(writing_list)
file.close()




