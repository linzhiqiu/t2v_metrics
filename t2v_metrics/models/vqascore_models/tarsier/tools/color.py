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
class Color:

    @staticmethod
    def red(x):
        return '\33[31m' +x + '\033[0m'
    
    @staticmethod
    def green(x):
        return '\33[32m' +x + '\033[0m'

    @staticmethod
    def yellow(x):
        return '\33[33m' +x + '\033[0m'

    @staticmethod
    def blue(x):
        return '\33[34m' +x + '\033[0m'

    @staticmethod
    def violet(x):
        return '\33[35m' +x + '\033[0m'

    
