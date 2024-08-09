from comet import download_model, load_from_checkpoint
import sys
import os

# en is excluded
language_sequence = ["af", "am", "ar", "as", "az", "be", "bg", "bn", "br",
                     "bs", "ca", "cs", "cy", "da", "de", "el", "nb", "eo",
                     "es", "et", "eu", "fa", "fi", "fr", "fy", "ga", "gd",
                     "gl", "gu", "ha", "he", "hi", "hr", "hu", "id", "is",
                     "it", "ja", "ka", "kk", "km", "kn", "ko", "ku", "ky",
                     "lt", "lv", "mg", "mk", "ml", "mr", "ms", "my", "ne",
                     "nl", "no", "or", "pa", "pl", "ps", "pt", "ro", "ru",
                     "si", "sk", "sl", "sq", "sr", "sv", "ta", "te", "th", 
                     "tr", "ug", "uk", "ur", "uz", "vi", "xh", "yi", "zh", "nn",]
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

def _compute_comet_(model, result_path, method_name, model_id, src, tgt):
    refs = _read_txt_strip_(os.path.join(result_path, str(method_name), str(model_id), f"{src}-{tgt}.r"))
    srcs = _read_txt_strip_(os.path.join(result_path, str(method_name), str(model_id), f"{src}-{tgt}.s"))
    hypos = _read_txt_strip_(os.path.join(result_path, str(method_name), str(model_id), f"{src}-{tgt}.h"))
    data = [{"src": src_text, "mt": mt_text, "ref": ref_text} for src_text, mt_text, ref_text in zip(srcs, hypos, refs)]
    model_output = model.predict(data, batch_size=200, gpus=1)
    score = round(model_output.system_score * 100, 2)
    return score

method_name = sys.argv[1]
model_id = sys.argv[2]
# 2en, en2, zero
flag = sys.argv[3]
work_path = sys.argv[4]
result_path = os.path.join(work_path, "results")

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

writing_list = []
if flag == "2en":
    tgt = "en"
    for src in language_sequence:
        score =  _compute_comet_(model, result_path, method_name, model_id, src, tgt)
        writing_list.append("{}-{}\n".format(src, tgt))
        writing_list.append(f"Score: {score} \n")
elif flag == "en2":
    src = "en"
    for tgt in language_sequence:
        score =  _compute_comet_(model, result_path, method_name, model_id, src, tgt)
        writing_list.append("{}-{}\n".format(src, tgt))
        writing_list.append(f"Score: {score} \n")
elif flag == "zero":
    for pair in zero_pairs:
        src, tgt = pair[0], pair[1]
        score =  _compute_comet_(model, result_path, method_name, model_id, src, tgt)
        writing_list.append("{}-{}\n".format(src, tgt))
        writing_list.append(f"Score: {score} \n")
        tgt, src = pair[0], pair[1]
        score =  _compute_comet_(model, result_path, method_name, model_id, src, tgt)
        writing_list.append("{}-{}\n".format(src, tgt))
        writing_list.append(f"Score: {score} \n")

file = open(os.path.join(result_path, str(method_name), str(model_id), "{}.comet".format(str(model_id))), 'a', encoding='utf-8')
file.writelines(writing_list)
file.close()
