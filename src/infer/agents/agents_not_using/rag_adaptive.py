from .base import BaseExpert
import jsonlines
import json
import copy
import re
import os
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import get_config
from langchain_core.prompts import ChatPromptTemplate
# --------------------------------------------------
#### 참고: https://github.com/AkariAsai/self-rag ####


# RAG prompts for all/partial modes
rag_prompt = ChatPromptTemplate.from_template("""
당신은 농업 전문 상담사입니다. 아래 참고 자료를 바탕으로 질문에 답변하세요.

**답변 작성 규칙:**
1. 참고 자료의 내용을 **이해하고 재구성**하여 자연스럽고 명확하게 설명하세요
2. 답변은 **핵심 정보만 간결하게**, 불필요한 배경설명·문장 반복을 피하세요  
3. 문장은 **짧고 직관적**으로 작성하고, 장문·장황한 서술을 하지 마세요
4. 원문을 그대로 복사하거나 표/목록 형식을 그대로 옮기지 마세요
5. 문서 출처, 페이지 번호, 파일명 등은 언급하지 마세요

---
[참고 자료]
{reviews}

---
[질문]
{question}

[답변]
""")

rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]",
                        "[Retrieval]", "[Continue to Use Evidence]"]
utility_tokens_names = ["[Utility:1]", "[Utility:2]",
                        "[Utility:3]", "[Utility:4]", "[Utility:5]"]
ground_tokens_names = ["[Fully supported]",
                    "[Partially supported]", "[No support / Contradictory]"]
other_special_tokens = ["<s>", "</s>", "[PAD]",
                        "<unk>", "<paragraph>", "</paragraph>"]
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]

adaptive_inst = (
    "Answer the following question. The question may be ambiguous and have multiple correct answers,\n"
    "and in that case, you have to provide a long-form answer including all correct answers."
)

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_open_instruct": (
        "<user>\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_open_instruct_retrieval": (
        "<user>\nReference:{paragraph}\n{instruction}\n"
        "<assistant>\n"
    ),
    "llama_chat_prompt": (
        "[INST]{instruction}[/INST]"
    ),
    "llama_chat_prompt_retrieval": (
        "[INST]{paragraph}\n{instruction}[/INST]"
    ),
}

class AdaptiveExpert(BaseExpert):
    """완전 vLLM 기반 Adaptive gating + RAG 전문가"""
    model_dir = os.getenv("MODEL_DIR", "./selfrag_llama2_7b")

    def setup(self, retriever_mode: str) -> None:
        self.model_dir = os.getenv("MODEL_DIR", "./selfrag_llama2_7b")
        self.model_dir = os.path.expanduser(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True, local_files_only=True, repo_type="model"
        )
        self.critic_llm = LLM(
            model=self.model_dir,
            model_impl="transformers",
            trust_remote_code=True,
            dtype="half",
            gpu_memory_utilization=0.9,
            max_num_seqs=1,
            max_model_len=1824
        )
        self.rag_tpl = rag_prompt
        self.args = get_config()
        self.gate_sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=25, logprobs=20)
        self.ans_sampling  = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256, logprobs=None)

        if retriever_mode not in self.retriever_map:
            raise ValueError(f"Unknown retriever_mode: {retriever_mode}")
        self.retriever = self.retriever_map[retriever_mode]

        (self.ret_tokens, self.rel_tokens, self.grd_tokens, self.ut_tokens) = load_special_tokens(
            self.tokenizer, use_grounding=True, use_utility=True
        )
        if self.ret_tokens:
            self.special_token_map = {
                "ret_tokens": self.ret_tokens,
                "rel_tokens": self.rel_tokens,
                "grd_tokens": self.grd_tokens,
                "ut_tokens": self.ut_tokens,
            }

    def _to_float(self, lp, default: float = -5.0) -> float:
        val = getattr(lp, "logprob", lp)
        try:
            return float(val)
        except:
            return default

    def _call_adaptive(self, prompt: str, evidences: list):
        a = self.args
        out = self.critic_llm.generate([prompt], self.gate_sampling, use_tqdm=False)[0]
        lp0    = out.outputs[0].logprobs[0]
        ret_lp = self._to_float(lp0.get(self.ret_tokens["[Retrieval]"]))
        no_lp  = self._to_float(lp0.get(self.ret_tokens["[No Retrieval]"]))
        p_ret  = float(np.exp(ret_lp))
        p_no   = float(np.exp(no_lp))
        gate_p = p_ret / (p_ret + p_no + 1e-12)

        if gate_p <= a.threshold:
            return prompt, {"gating": {"p_ret": p_ret, "p_no": p_no, "used": False}}, False

        if evidences:
            best = evidences[0]
            final_prompt = (
                prompt + f"[Retrieval]<paragraph>{best['title']}\n{best['text']}</paragraph>\n\nAnswer:"
            )
            return final_prompt, {"gating": {"used": True}}, True

        return prompt, {"gating": {"used": False, "no_evidence": True}}, False

    def handle(self, question: str):
        prompt0 = PROMPT_DICT["prompt_no_input"].format_map({"instruction": question})
        docs    = self.retriever.get_relevant_documents(question)
        ev      = [{"title": d.metadata.get("source",""), "text": d.page_content} for d in docs]

        final_prompt, info, used = self._call_adaptive(prompt0, ev)
        if used:
            prompt_text = self.rag_tpl.format(reviews=ev[0]["text"], question=question)
        else:
            prompt_text = prompt0 + "\n\nAnswer:"

        outputs = self.critic_llm.generate([prompt_text], self.ans_sampling)
        answer = outputs[0].outputs[0].text.strip()
        return answer, info
        
    def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
        ret_tokens = {token: tokenizer.convert_tokens_to_ids(
            token) for token in retrieval_tokens_names}
        rel_tokens = {}
        for token in ["[Irrelevant]", "[Relevant]"]:
            rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        grd_tokens = None
        if use_grounding is True:
            grd_tokens = {}
            for token in ground_tokens_names:
                grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        ut_tokens = None
        if use_utility is True:
            ut_tokens = {}
            for token in utility_tokens_names:
                ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        return ret_tokens, rel_tokens, grd_tokens, ut_tokens


    def fix_spacing(input_text):
        # Add a space after periods that lack whitespace
        output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
        return output_text


    def load_file(input_fp):
        def load_jsonlines(file):
            with jsonlines.open(file, 'r') as jsonl_f:
                lst = [obj for obj in jsonl_f]
            return lst
        if input_fp.endswith(".json"):
            input_data = json.load(open(input_fp))
        else:
            input_data = load_jsonlines(input_fp)
        return input_data

    def save_file_jsonl(data, fp):
        with jsonlines.open(fp, mode='w') as writer:
            writer.write_all(data)

    def preprocess_input(input_data, task):
        if task == "factscore":
            for item in input_data:
                item["instruction"] = item["input"]
                item["output"] = [item["output"]
                                ] if "output" in item else [item["topic"]]
            return input_data

        elif task == "qa":
            for item in input_data:
                if "instruction" not in item:
                    item["instruction"] = item["question"]
                if "answers" not in item and "output" in item:
                    item["answers"] = "output"
            return input_data

        elif task in ["asqa", "eli5"]:
            processed_input_data = []
            for instance_idx, item in enumerate(input_data["data"]):
                prompt = item["question"]
                instructions = adaptive_inst
                prompt = instructions + "## Input:\n\n" + prompt
                entry = copy.deepcopy(item)
                entry["instruction"] = prompt
                processed_input_data.append(entry)
            return processed_input_data


    def postprocess_output(input_instance, prediction, task, intermediate_results=None):
        def postprocess(pred):
            special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                            "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
            for item in special_tokens:
                pred = pred.replace(item, "")
            pred = pred.replace("</s>", "")

            if len(pred) == 0:
                return ""
            if pred[0] == " ":
                pred = pred[1:]
            return pred

        if task == "factscore":
            return {"input": input_instance["input"], "output": prediction, "topic": input_instance["topic"], "cat": input_instance["cat"]}

        elif task == "qa":
            input_instance["pred"] = prediction
            return input_instance

        elif task in ["asqa", "eli5"]:
            # ALCE datasets require additional postprocessing to compute citation accuracy.
            final_output = ""
            docs = []
            if "splitted_sentences" not in intermediate_results:
                input_instance["output"] = postprocess(prediction)

            else:
                for idx, (sent, doc) in enumerate(zip(intermediate_results["splitted_sentences"][0], intermediate_results["ctxs"][0])):
                    if len(sent) == 0:
                        continue
                    postprocessed_result = postprocess(sent)
                    final_output += postprocessed_result[:-
                                                        1] + " [{}]".format(idx) + ". "
                    docs.append(doc)
                if final_output[-1] == " ":
                    final_output = final_output[:-1]
                input_instance["output"] = final_output
            input_instance["docs"] = docs
            return input_instance

    def process_arc_instruction(item, instruction):
        choices = item["choices"]
        answer_labels = {}
        for i in range(len(choices["label"])):
            answer_key = choices["label"][i]
            text = choices["text"][i]
            if answer_key == "1":
                answer_labels["A"] = text
            if answer_key == "2":
                answer_labels["B"] = text
            if answer_key == "3":
                answer_labels["C"] = text
            if answer_key == "4":
                answer_labels["D"] = text
            if answer_key in ["A", "B", "C", "D"]:
                answer_labels[answer_key] = text

        if "D" not in answer_labels:
            answer_labels["D"] = ""
        choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
        if "E" in answer_labels:
            choices += "\nE: {}".format(answer_labels["E"])
        processed_instruction = instruction + "\n\n### Input:\n" + item["instruction"] + choices
        return processed_instruction


    def postprocess_answers_closed(output, task, choices=None):
        final_output = None
        if choices is not None:
            for c in choices.split(" "):
                if c in output:
                    final_output = c
        if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
            final_output = "true" if output == "SUPPORTS" else "REFUTES"
        if task == "fever" and output.lower() in ["true", "false"]:
            final_output = output.lower()
        if final_output is None:
            return output
        else:
            return final_output
    