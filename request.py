from dataclasses import dataclass
from typing import List
from enum import Enum

class AccLenGroundTruth(Enum):
    NGRAM = 1
    EAGLE = 2
    EAGLE_THREE = 3
    COMBINE_NGRAM_EAGLE = 4
    COMBINE_NGRAM_EAGLE_THREE = 5

@dataclass
class Request:
    id: str
    method: AccLenGroundTruth
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    acc_lens_ngram: list[int]
    acc_lens_eagle: list[int]
    acc_lens_eagle_three: list[int]

    probs_eagle_three: list[float] = None
    entropy_eagle_three: list[float] = None

    cur_gen_len: int = 0

    @classmethod
    def from_json(cls, json_data, method):
        return cls(
            id=json_data['id'],
            method=method,
            prompt_token_ids=json_data['prompt_token_ids'],
            generated_token_ids=json_data['generated_token_ids'],
            acc_lens_ngram=json_data['acc_ngram'].get('acc_len', None) if 'acc_ngram' in json_data else json_data['acc'].get('acc_len', None),
            acc_lens_eagle=json_data['acc_eagle'].get('acc_len', None) if 'acc_eagle' in json_data else json_data['acc'].get('acc_len', None),
            acc_lens_eagle_three=json_data.get('acc_eagle_three', None),
        )

    @property
    def prompt_len(self):
        return len(self.prompt_token_ids)

    @property
    def generated_len(self):
        if self.method == AccLenGroundTruth.NGRAM:
            assert self.acc_lens_ngram is not None
            return len(self.acc_lens_ngram)
        elif self.method == AccLenGroundTruth.EAGLE:
            assert self.acc_lens_eagle is not None
            return len(self.acc_lens_eagle)
        elif self.method == AccLenGroundTruth.EAGLE_THREE:
            assert self.acc_lens_eagle_three is not None
            return len(self.acc_lens_eagle_three)
        elif self.method == AccLenGroundTruth.COMBINE_NGRAM_EAGLE:
            assert self.acc_lens_ngram is not None
            # NOTE: this is not necessarily correct
            return min(len(self.acc_lens_ngram), len(self.acc_lens_eagle))
        elif self.method == AccLenGroundTruth.COMBINE_NGRAM_EAGLE_THREE:
            assert self.acc_lens_ngram is not None
            return min(len(self.acc_lens_ngram), len(self.acc_lens_eagle_three))
        else:
            raise ValueError(f"Unknown Acc Len Ground Truth Method: {self.method}")

    @property
    def finished(self):
        return self.cur_gen_len >= self.generated_len