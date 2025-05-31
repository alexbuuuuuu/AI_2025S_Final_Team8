import pandas as pd
import os
import json
import re


def calculate_binary_answer(text: str, threshold: int=70) -> int:
    """計算文本的二元答案（1或0）"""
    # 假設這裡有一個簡單的邏輯來決定答案
    # 例如，如果文本包含某些關鍵詞則返回1，否則返回0
    regex = r'違法機率[:：]\s*(\d+)%'
    prob = re.search(regex, text)
    # print(f"檢測到違法機率: {prob.group(1) if prob else '無'}")
    if prob is not None and int(prob.group(1)) >= threshold:
        return 0
    else:
        return 1
    
    


if not os.path.exists('./diff_thresholds'):
    os.makedirs('./diff_thresholds', exist_ok=True)

with open('./results.json', 'r') as f:
    results = json.load(f)


print("Loaded results from results.json")
for i in range(0, 101, 5):
    print(f"Processing threshold: {i}")
    answers =  []
    for result in results:
        text = result["response"]
        answer = calculate_binary_answer(text, threshold=i)
        answers.append(answer)
        
    # 生成指定格式的 Kaggle 結果
    answer_frame = pd.DataFrame({ "ID": [ i for i in range(0, len(answers))], "Answer": answers })
    answer_frame.to_csv(f"./diff_thresholds/diff_threshold_{i}.csv", index=False)