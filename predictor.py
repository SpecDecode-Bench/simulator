from enum import Enum
from request import Request

class Llama8BH100:
    # fitted with instructcoder bs=1,8,16,32,64
    target_c_kv = 3.92645679e-08
    target_c_compute = 2.10103129e-05
    target_c_fixed = 5.88590713e-03

    draft_c_kv = 2.37768493e-09
    draft_c_num_spec_tokens = 8.41429603e-07
    draft_c_fixed = 2.13526175e-03

    ngram_draft_percentage = 0.005875
    overhead_constant_per_step = 1.20683946e-03
    combined_switching_overhead = 0.0


class ProposeMethod(Enum):
    NGRAM = 1
    EAGLE = 2

class AccLenPredictMethod(Enum):
    FIXED_1 = 1
    FIXED_3 = 2
    FIXED_5 = 3
    ORACLE = 4

ENTROPY_THRESHOLD = 5
class TimePredictor:
    def __init__(self, method: ProposeMethod):
        self.model = Llama8BH100
        self.method = method

    def predict_forward_pass_time(self, num_tokens_in_kv_cache: int, num_batched_tokens: int):
        return num_tokens_in_kv_cache * self.model.target_c_kv + num_batched_tokens * self.model.target_c_compute + self.model.target_c_fixed

    def get_overhead_per_request(self):
        print("Warning: this should not be used.")
        return self.model.overhead_constant

    def predict_draft_time(self, verify_time, num_tokens_in_kv_cache: int, num_spec_tokens: int):
        if self.method == ProposeMethod.NGRAM:
            return verify_time * self.model.ngram_draft_percentage
        elif self.method == ProposeMethod.EAGLE:
            return num_tokens_in_kv_cache * self.model.draft_c_kv + num_spec_tokens * self.model.draft_c_num_spec_tokens + self.model.draft_c_fixed
        else:
            raise ValueError(f"Unsupported draft method: {self.method}")

    def get_overhead_per_step(self):
        return self.model.overhead_constant_per_step

    def get_switching_overhead(self):
        return self.model.combined_switching_overhead

class AccLenPredictor:
    def __init__(self, method: AccLenPredictMethod):
        self.model = method

    def predict(self,
                req: Request,
                gt_acc_len: int,
                ) -> int:
        if self.model == AccLenPredictMethod.FIXED_1:
            return 2
        elif self.model == AccLenPredictMethod.FIXED_3:
            return 4
        elif self.model == AccLenPredictMethod.FIXED_5:
            return 6
        elif self.model == AccLenPredictMethod.ORACLE:
            return gt_acc_len
        else:
            raise ValueError(f"Unknown method: {self.model}")

    def _get_acc_len(self, probs, threshold):
        acc_len = 0
        for i, prob in enumerate(probs):
            if prob > threshold:
                acc_len += 1
            else:
                break

        return acc_len
