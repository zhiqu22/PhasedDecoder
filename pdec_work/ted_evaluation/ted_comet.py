from comet import download_model, load_from_checkpoint
import sys
import os

language_sequence = ["en", "ar", "he", "ru", "ko", "it", "ja", "zh", "es", "nl", "vi", "tr", "fr", "pl", "ro", "fa", "hr", "cs", "de"]

def _read_txt_strip_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]

method_name = sys.argv[1]
model_id = sys.argv[2]
tgt  = sys.argv[3]
save_path = sys.argv[4]
save_path = os.path.join(save_path, "results")
model_path = download_model("Unbabel/wmt22-comet-da")

model = load_from_checkpoint(model_path)

writing_list = []
for src in language_sequence:
    if src == tgt: continue
    refs = _read_txt_strip_(os.path.join(save_path, str(method_name), str(model_id), "{}-{}.detok.r".format(src, tgt)))
    hypos = _read_txt_strip_(os.path.join(save_path, str(method_name), str(model_id), "{}-{}.detok.h".format(src, tgt)))
    srcs = _read_txt_strip_(os.path.join(save_path, str(method_name), str(model_id), "{}-{}.detok.s".format(src, tgt)))
    data = [{"src": src_text, "mt": mt_text, "ref": ref_text} for src_text, mt_text, ref_text in zip(srcs, hypos, refs)]
    model_output = model.predict(data, batch_size=200, gpus=1)
    score = round(model_output.system_score * 100, 2)
    writing_list.append(f"{src}-{tgt}\n")
    writing_list.append(f"Score: {score} \n")

file = open(os.path.join(save_path, str(method_name), str(model_id), "{}.comet".format(str(model_id))), 'a', encoding='utf-8')
file.writelines(writing_list)
file.close()
