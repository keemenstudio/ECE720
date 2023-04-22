import json

with open('pgd_result.json') as f:
    json_obj = json.load(f)
    success_cnt = 0
    snr = 0
    cc = 0
    for line in json_obj:
        if line['as'] == 1:
            success_cnt += 1
            snr += float(line['snr'])
            cc += float(line['cc'])
    print(f"success rate {success_cnt/len(json_obj)}")
    print(f"snr {snr/success_cnt}")
    print(f"cc {cc/success_cnt}")