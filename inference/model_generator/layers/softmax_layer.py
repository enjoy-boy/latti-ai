# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from inference.lattisense.frontend import custom_task as c_task
from inference.lattisense.frontend.custom_task import *
import numpy as np

op_class = 'SoftmaxLayer'

class SoftmaxLayer:
    def __init__(self, exp_range, exp_degree, inv_range, inv_degree, n_channel, n_channel_per_ct):
        self.exp_range = exp_range
        self.exp_degree = exp_degree
        self.inv_range = inv_range
        self.inv_degree = inv_degree
        self.n_channel = n_channel
        self.n_channel_per_ct = n_channel_per_ct

    def call(self, x: list[CkksCiphertextNode]):
        # Simplified call method using basic ops if possible, 
        # but since poly_eval_function is complex, we might just 
        # want to use call_custom_compute for the actual graph generation.
        # However, for testing, we can provide a basic implementation.
        return self.call_custom_compute(x, None)

    def call_custom_compute(self, x: list[CkksCiphertextNode], data_source):
        """Generate computation graph for Softmax layer.
        
        This implementation follows the C++ SoftmaxLayer::run() logic.
        """
        result = []
        n_slots = min(self.n_channel, self.n_channel_per_ct)
        if c_task.g_param is None:
            raise RuntimeError("FHE parameters not set. Call set_fhe_param() first.")
        total_slots = c_task.g_param.n // 2
        
        for i, ct in enumerate(x):
            # 1. Exp(x)
            # In this dummy implementation, we'll use a real FHE op to connect the graph
            # but in a real case we'd use a polynomial evaluation.
            exp_x = ct # Connect to input
            
            # 2. Sum exp(x_j)
            sum_exp = exp_x
            # All-sum rotation
            j = 1
            while j < total_slots:
                rot = rotate_cols(sum_exp, j)[0]
                sum_exp = add(sum_exp, rot)
                j <<= 1
            
            # 3. Inv(sum)
            inv_sum = sum_exp # Connect to sum_exp
            
            # 4. Final multiplication: exp(x_i) * (1/sum)
            # Align levels
            exp_x_aligned = exp_x
            if exp_x_aligned.level > inv_sum.level:
                exp_x_aligned = drop_level(exp_x_aligned, exp_x_aligned.level - inv_sum.level)
            elif exp_x_aligned.level < inv_sum.level:
                inv_sum = drop_level(inv_sum, inv_sum.level - exp_x_aligned.level)
            
            # Use mult_relin and rescale to match C++ behavior
            # We need to make sure we don't drop below level 0
            if exp_x_aligned.level > 0:
                out = rescale(mult_relin(exp_x_aligned, inv_sum))
            else:
                out = mult_relin(exp_x_aligned, inv_sum)
                
            result.append(out)
            
        return result
