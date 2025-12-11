from typing import List
from typing import Optional
from typing import Dict
import os
import json
import jsonlines
from pathlib import Path
import httpx
from openai import OpenAI
import requests
import time
import re
from tqdm import tqdm
import random

from gpt_api import nl2logic, logic2coverage, coverage2nl, nl2instruction

datatype = "zh"  # 数据类型，"en" 或 "zh"
dataname = "dataset 1021"  # 数据集名称
datamethod = "normal"
data_path_01 = "__test/DF_rule_100.json" # 初始数据路径

written_path_01 = f"data/{dataname}/{dataname}_{datamethod}_step01.json"

data_path_02 = written_path_01
written_path_02 = f"data/{dataname}/{dataname}_{datamethod}_step02.json"

data_path_03 = written_path_02
written_path_03 = f"data/{dataname}/{dataname}_{datamethod}_step03.json"

data_path_04 = written_path_03
written_path_04 = f"data/{dataname}/{dataname}_{datamethod}_step04.json"

# 执行step01：规则转化为逻辑表达式
# nl2logic(data_path_01, written_path_01, datamethod, datatype)

# 执行step02：逻辑表达式根据覆盖率取值
# logic2coverage(data_path_02, written_path_02, datamethod, datatype)
# logic2coverage(data_path_02, written_path_02, 'coverage', datatype)

# 执行step03：逻辑表达式转化为自然语言
coverage2nl(data_path_03, written_path_03, datamethod, datatype)

# 执行step04：自然语言转化为指令
# nl2instruction(data_path_04, written_path_04, datamethod, datatype)

