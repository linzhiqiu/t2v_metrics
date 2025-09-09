# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from json import JSONEncoder
import numpy
import pandas as pd

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def write_txt(data, path):
    with open(path, 'w', encoding='utf-8')as f:
        for d in data:
            f.write(f'{d}\n')

def read_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip('\n') for l in f.readlines()]
        return lines

def read_jsonlines(path):
    objs = []
    with open(path) as f:
        for line in f:
            line = json.loads(line)
            objs.append(line)
    return objs

def write_jsonlines(data, path, cls=None, ensure_ascii=False):
    with open(path, 'w') as f:
        for d in data:
            d = json.dumps(d, ensure_ascii=ensure_ascii, cls=cls)
            f.write(d)
            f.write('\n')

def read_parquet(path):
    data = pd.read_parquet(path)
    return data.to_dict('records')

def write_parquet(data, path):
    data = pd.DataFrame(data)
    data.to_parquet(path)

def read_csv(path):
    data = pd.read_csv(path)
    return data.to_dict(orient='records')

def write_csv(data, path):
    data = pd.DataFrame(data)
    data.to_csv(path, index=False, sep='\t')
