# 双向标签变换

label_to_index = {
    "早疫病": 0,
    "健康": 1,
    "晚疫病": 2
}

index_to_label = dict((v,k) for k, v in label_to_index.items())
