import json

# 读取JSON文件
with open('/home/xke001/demo/zero-shot/datasets/trials/object_5_3_42.json', 'r') as file:
    data = json.load(file)

# 保留前三个元素
first_three_items = data[:10]

# 将更新后的数据写回到文件
with open('example.json', 'w') as file:
    json.dump(first_three_items, file, indent=4)

print("Updated JSON file with the first three items.")