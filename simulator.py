import json
import random
import copy
from request import Request, AccLenGroundTruth
from predictor import (TimePredictor,
                       AccLenPredictor,
                       AccLenPredictMethod,
                       ProposeMethod)
from tqdm import tqdm

NGRAM_MATCHING_PROBABILITY = {
    "sharegpt": 0.2,
    "cnn": 0.4,
    "gsm8k": 0.2,
    "instructcoder": 0.8, # a higher matched prob
}
SEED = 42
random.seed(SEED)
class SpecSimulator:
    def __init__(self, file_path: str,
                 acc_predict_method: AccLenPredictMethod,
                 acc_gt_method: AccLenGroundTruth,
                 dataset: str):
        self.file_path = file_path
        self.data = self.load_data(acc_gt_method)
        self.acc_gt_method = acc_gt_method
        self.acc_predict_method = acc_predict_method
        self.dataset = dataset

        self.time_predictor = TimePredictor(
            method=self.get_propose_method(),
        )
        self.acc_len_predictor = AccLenPredictor(acc_predict_method)


    def get_propose_method(self):
        if self.acc_gt_method == AccLenGroundTruth.NGRAM:
            return ProposeMethod.NGRAM
        elif self.acc_gt_method == AccLenGroundTruth.EAGLE or self.acc_gt_method == AccLenGroundTruth.EAGLE_THREE:
            return ProposeMethod.EAGLE
        elif self.acc_gt_method == AccLenGroundTruth.COMBINE_NGRAM_EAGLE or self.acc_gt_method == AccLenGroundTruth.COMBINE_NGRAM_EAGLE_THREE:
            print(f"[Warning] Combined method detected, effective drafting method will be determined at runtime.")
            return ProposeMethod.NGRAM
        else:
            print(f"[Warning] Unknown Ground Truth Acc Len: {self.acc_gt_method}, defaulting to use NGRAM.")
            return ProposeMethod.NGRAM

    def load_data(self, method: AccLenGroundTruth) -> list[Request]:
        data: list[Request] = []
        with open(self.file_path, 'r') as file:
            for i, line in enumerate(file):
                d = Request.from_json(json.loads(line), method)
                if not d.finished:
                    data.append(d)
        return data

    def _dup_request(self, req: Request, new_id: str) -> Request:
        new_req = copy.deepcopy(req)
        new_req.id = new_id
        return new_req

    def simulate_like_experiment(self, batch_size=1):
        random.seed(SEED)
        requests = random.sample(self.data, k=300)

        batch_speedups = []
        batch_org_latency = []
        batch_sd_latency = []
        batch_org_tpt = []
        batch_sd_tpt = []
        for req in tqdm(requests, total=len(requests) ,desc=f"Simulating batches of size {batch_size}"):
            batch_data = []
            for i in range(batch_size):
                batch_data.append(self._dup_request(req, f"{req.id}-{i}"))
            org_latencies, org_verify_times = self.simulate_org_batch(batch_data)
            sd_latencies, sd_verify_times, sd_propose_times = self.simulate_sd_batch(batch_data)
            speedups = {}
            request_org_latencies = {}
            request_sd_latencies = {}
            tokens_generated = 0
            for req in batch_data:
                tokens_generated += req.generated_len
                org_latency = sum(org_latencies[req.id])
                sd_latency = sum(sd_latencies[req.id])
                speedups[req.id] = org_latency / sd_latency
                request_org_latencies[req.id] = org_latency
                request_sd_latencies[req.id] = sd_latency

            org_duration = sum(request_org_latencies.values()) / len(request_org_latencies)
            sd_duration = sum(request_sd_latencies.values()) / len(request_sd_latencies)

            batch_org_tpt.append(tokens_generated / org_duration)
            batch_sd_tpt.append(tokens_generated / sd_duration)

            batch_org_latency.append(org_duration)
            batch_sd_latency.append(sd_duration)

            batch_speedups.append(sum(speedups.values()) / len(speedups))
        avg_speedup = sum(batch_speedups) / len(batch_speedups)

        avg_org_latency = sum(batch_org_latency) / len(batch_org_latency)
        avg_sd_latency = sum(batch_sd_latency) / len(batch_sd_latency)
        assert len(batch_org_latency) == len(batch_sd_latency)
        avg_speedup_by_request_latency = sum(batch_org_latency)/sum(batch_sd_latency)

        # Throughput
        avg_org_tpt = sum(batch_org_tpt) / len(batch_org_tpt)
        avg_sd_tpt = sum(batch_sd_tpt) / len(batch_sd_tpt)
        avg_speedup_by_tpt = avg_sd_tpt / avg_org_tpt

        print(f"---BS={batch_size}---")
        print(f"Average org tpt: {avg_org_tpt:.2f}, average sd tpt: {avg_sd_tpt:.2f}, speedup by tpt: {avg_speedup_by_tpt:.2f}")
        print(f"Average org latency per request: {avg_org_latency:.2f}, average sd latency per request: {avg_sd_latency:.2f}, speedup by request latency: {avg_speedup_by_request_latency:.2f}")
        print(f"Average speedup for batch size {batch_size}: {avg_speedup:.2f}")
        print(f"------")
        return avg_speedup_by_tpt

    def simulate_org_batch(self, data: list[Request]) -> dict[str, list[float]]:
        # -1 here to exclude the prefill token.
        max_decode_len = max([req.generated_len - 1 for req in data])
        num_prompt_tokens = sum([req.prompt_len for req in data])
        num_tokens_in_kv = 0

        # for debugging-only
        verify_times = []

        # Prefill
        prefill_time = self.time_predictor.predict_forward_pass_time(num_tokens_in_kv, num_prompt_tokens)
        verify_times.append(prefill_time)
        num_tokens_in_kv += num_prompt_tokens + 1*len(data)
        request_latencies = {}
        for req in data:
            request_latencies[req.id] = [prefill_time, self.time_predictor.get_overhead_per_step()]

        # Decode
        for step in range(max_decode_len):
            num_batched_tokens = 0
            for req in data:
                if step < req.generated_len - 1:
                    num_batched_tokens += 1

            decode_time = self.time_predictor.predict_forward_pass_time(num_tokens_in_kv, num_batched_tokens)
            num_tokens_in_kv += num_batched_tokens
            verify_times.append(decode_time)

            for req in data:
                if step < req.generated_len - 1:
                    request_latencies[req.id].append(decode_time + self.time_predictor.get_overhead_per_step())
                if step == req.generated_len - 2: # request is finished, release kv cache
                    num_tokens_in_kv = num_tokens_in_kv - (req.generated_len) - (req.prompt_len)
                    assert num_tokens_in_kv >= 0

        assert num_tokens_in_kv == 0
        return request_latencies, verify_times


    def simulate_sd_batch(self, data: list[Request]) -> dict[str, float]:
        unfinished_requests = copy.copy(data)
        num_prompt_tokens = sum([req.prompt_len for req in data])
        num_tokens_in_kv = 0

        # for debugging only
        verify_times = []
        propose_times = []

        # Prefill
        prefill_time = self.time_predictor.predict_forward_pass_time(num_tokens_in_kv, num_prompt_tokens)
        verify_times.append(prefill_time)

        # + 1 for the first token generated
        num_tokens_in_kv += num_prompt_tokens + 1*len(data)
        request_latencies = {}
        for req in data:
            request_latencies[req.id] = [prefill_time, self.time_predictor.get_overhead_per_step()]
            req.cur_gen_len = 1

        step_input_lens = {}
        prev_winning_method = {}
        step = 0
        while len(unfinished_requests) > 0:
            num_batched_tokens = 0
            draft_time = {}
            for req in unfinished_requests:
                input_len = self.get_input_len(req)
                step_input_lens[req.id] = input_len
                num_batched_tokens += step_input_lens[req.id]

            verify_time = self.time_predictor.predict_forward_pass_time(num_tokens_in_kv, num_batched_tokens)
            verify_times.append(verify_time)

            # Calculate max draft time across all requests in this step (proposal happens once per step)
            max_draft_time = 0
            unfinished_requests_after_step = []
            for req in unfinished_requests:
                # For combined methods, determine the winning drafter first so draft time is accurate
                step_switching_overhead = 0.0
                effective_draft_method = None
                if self.acc_gt_method in (AccLenGroundTruth.COMBINE_NGRAM_EAGLE, AccLenGroundTruth.COMBINE_NGRAM_EAGLE_THREE):
                    cur_winner = self._get_winning_method(req)
                    effective_draft_method = cur_winner
                    if req.id in prev_winning_method and prev_winning_method[req.id] != cur_winner:
                        step_switching_overhead = self.time_predictor.get_switching_overhead()
                    prev_winning_method[req.id] = cur_winner

                # ngram not matched, NOTE: not very accurate, it could be matched but not accepted.
                if self.acc_gt_method == AccLenGroundTruth.NGRAM and step_input_lens[req.id] == 1:
                    draft_time = 0
                else:
                    draft_time = self.time_predictor.predict_draft_time(verify_time, num_tokens_in_kv, step_input_lens[req.id], method=effective_draft_method)
                max_draft_time = max(max_draft_time, draft_time)

                request_latencies[req.id].append(verify_time + draft_time + self.time_predictor.get_overhead_per_step() + step_switching_overhead)
                step_gen_len = self.get_step_gen_len(req)
                accepted_tokens = min(step_gen_len, step_input_lens[req.id])
                # only update adaptive k if there are more than 1 draft tokens; handle the case where ngram is not matched
                if step_input_lens[req.id] > 1:
                    self.acc_len_predictor.update(req.id, step_input_lens[req.id], accepted_tokens)
                req.cur_gen_len += accepted_tokens
                num_tokens_in_kv += accepted_tokens
                if not req.finished:
                    unfinished_requests_after_step.append(req)
                if req.finished:
                    num_tokens_in_kv = num_tokens_in_kv - req.cur_gen_len - req.prompt_len
                    assert num_tokens_in_kv >= 0
            propose_times.append(max_draft_time)
            unfinished_requests = unfinished_requests_after_step
            step += 1

        return request_latencies, verify_times, propose_times

    # Here, the input length
    # includes the last generation token.
    # input_len = proposed_len + 1
    def get_input_len(self, req: Request) -> int:
        # NGRAM is not matched, just return 1; This is just a very rough estimation
        if self.acc_gt_method == AccLenGroundTruth.NGRAM and self._not_matched_ngram(req):
            return 1

        gt_acc_len = self.get_step_gen_len(req)
        return self.acc_len_predictor.predict(req,
                                            gt_acc_len)

    def _not_matched_ngram(self, req: Request) -> bool:
        gt_acc_len = self.get_step_gen_len(req)
        # if orcale, there is always a match
        if self.acc_predict_method == AccLenPredictMethod.ORACLE:
            return False
        # else, we simulate a probability of not matching
        return gt_acc_len == 1 and not random.random() < NGRAM_MATCHING_PROBABILITY[self.dataset]

    def _get_winning_method(self, req: Request) -> ProposeMethod:
        """For combined proposers, return which sub-method produces the longer acceptance length at the current position."""
        pos = req.cur_gen_len - 1
        if self.acc_gt_method == AccLenGroundTruth.COMBINE_NGRAM_EAGLE:
            if req.acc_lens_ngram[pos] >= req.acc_lens_eagle[pos]:
                return ProposeMethod.NGRAM
            return ProposeMethod.EAGLE
        elif self.acc_gt_method == AccLenGroundTruth.COMBINE_NGRAM_EAGLE_THREE:
            if req.acc_lens_ngram[pos] >= req.acc_lens_eagle_three[pos]:
                return ProposeMethod.NGRAM
            return ProposeMethod.EAGLE
        return None

    def get_step_gen_len(self, req: Request) -> int:
        if self.acc_gt_method == AccLenGroundTruth.NGRAM:
            return req.acc_lens_ngram[req.cur_gen_len - 1]
        elif self.acc_gt_method == AccLenGroundTruth.EAGLE:
            return req.acc_lens_eagle[req.cur_gen_len - 1]
        elif self.acc_gt_method == AccLenGroundTruth.EAGLE_THREE:
            return req.acc_lens_eagle_three[req.cur_gen_len - 1]
        elif self.acc_gt_method == AccLenGroundTruth.COMBINE_NGRAM_EAGLE:
            return max(req.acc_lens_ngram[req.cur_gen_len - 1], req.acc_lens_eagle[req.cur_gen_len - 1])
        elif self.acc_gt_method == AccLenGroundTruth.COMBINE_NGRAM_EAGLE_THREE:
            return max(req.acc_lens_ngram[req.cur_gen_len - 1], req.acc_lens_eagle_three[req.cur_gen_len - 1])
        else:
            raise ValueError(f"Unknown Ground Truth Acc Len: {self.acc_gt_method}")