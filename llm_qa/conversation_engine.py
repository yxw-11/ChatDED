import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer

from data.data_loader import read_txt_files
from llm_qa.request_llm import dp_answer
from prompts.prompt import FIRST_ROUND, SECOND_ROUND, LAST_ROUND, SUMMARIZE_DATA, ANALYZE_DATA
from rag.chunk import get_top_k_similar_chunks


IRREGULAR_RESPON = """
Agent: The meibography image was analyzed, but the quality is classified as poor. This means the scan is too blurred or obstructed for accurate gland evaluation.

Agent: To ensure reliable results, I recommend re-capturing the image with better eyelid exposure and reduced blinking. If the quality remains poor after repeat attempts, I will flag this case for manual review by your doctor.

Agent: For now, I cannot generate trustworthy gland measurements. However, I have documented your symptoms of dryness and mild stinging, and these will be integrated with other clinical tests while we wait for a valid image.
"""


@dataclass
class DialogueState:
    """State container for one patient case conversation."""

    case_index: int
    symptom: str
    detail: str
    round_three: str
    conversation_lines: List[str] = field(default_factory=list)
    summary: str = ""
    image_response: str = ""
    rag_info: str = ""
    suggestions: str = ""

    def add_patient(self, text: str) -> None:
        self.conversation_lines.append(f"Patient: {text}")

    def add_agent(self, text: str) -> None:
        self.conversation_lines.append(f"Agent: {text}")

    def add_raw(self, text: str) -> None:
        self.conversation_lines.append(text)

    def render(self) -> str:
        return "\n".join(self.conversation_lines) + "\n"


class MultiTurnConversationEngine:
    """A state-driven multi-turn pipeline while preserving original IO behavior."""

    def __init__(self, df_total: pd.DataFrame, round_columns: pd.DataFrame, doc_path: str):
        self.df_total = df_total
        self.round_columns = round_columns

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model.max_seq_length = 512

        self.all_text = read_txt_files(doc_path)

    @staticmethod
    def fill_first_p(prompt: str, symptom: str) -> str:
        return prompt.format(symptom=symptom)

    @staticmethod
    def fill_second_p(prompt: str, response: str, detail: str) -> str:
        return prompt.format(respon=response, detail=detail)

    @staticmethod
    def fill_analyze_p(
        prompt: str,
        d_1,
        d_2,
        d_3,
        d_4,
        d_5,
        d_6,
        d_7,
        d_8,
        d_9,
    ) -> str:
        return prompt.format(
            image_quality=d_1,
            avg_length=d_2,
            avg_width=d_3,
            avg_area=d_4,
            curvature=d_5,
            gland_count=d_6,
            total_prop=d_7,
            dropout_rate=d_8,
            grade=d_9,
        )

    @staticmethod
    def fill_sum_p(prompt: str, records: str) -> str:
        return prompt.format(records=records)

    @staticmethod
    def fill_last_p(prompt: str, conversation: str, rag_info: str) -> str:
        return prompt.format(conv=conversation, rag_info=rag_info)

    @staticmethod
    def retrieve_code(text: str) -> str:
        matches = re.findall(r"(?m)^Code:\s*(.+)$", text)
        return matches[-1] if matches else ""

    @staticmethod
    def run_code(code: str, df: pd.DataFrame):
        if not code:
            return ""
        try:
            return eval(code, {"df": df})
        except Exception:
            print("code execution error")
            return ""

    @staticmethod
    def get_row_vars_by_pos(df: pd.DataFrame) -> Tuple:
        cols = [
            "Qulity of meibography",
            "Average Length",
            "Average Width",
            "Average Area",
            "Average Curvature",
            "Number of glands",
            "Total Glands Ratio",
            "Glandular deletion ratio",
            "score",
        ]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        row = df.iloc[0]
        return tuple(row[c] for c in cols)

    def _fallback_patient_lookup(self, case_id: str) -> pd.DataFrame:
        """Keeps the original retrieval contract but adds a safe fallback."""
        if "ID" in self.df_total.columns:
            return self.df_total[self.df_total["ID"] == case_id]
        return pd.DataFrame()

    def _turn_opening(self, state: DialogueState) -> str:
        print("Patient: ", state.symptom)
        state.add_patient(state.symptom)
        first_prompt = self.fill_first_p(FIRST_ROUND, state.symptom)
        first_response = dp_answer(first_prompt)
        print("Agent: ", first_response)
        state.add_agent(first_response)
        return first_response

    def _turn_retrieve_record(self, state: DialogueState, first_response: str) -> Tuple[pd.DataFrame, str, str]:
        print("Patient: ", state.detail)
        state.add_patient(state.detail)

        case_id = f"P{state.case_index + 1}"
        second_prompt = self.fill_second_p(
            SECOND_ROUND,
            first_response,
            state.detail + f"Information: ID = {case_id}",
        )
        second_response = dp_answer(second_prompt)
        code = self.retrieve_code(second_response)

        retrieved_df = self.run_code(code, self.df_total)
        if isinstance(retrieved_df, str) or retrieved_df is None or len(retrieved_df) == 0:
            retrieved_df = self._fallback_patient_lookup(case_id)

        if len(retrieved_df) == 0:
            raise ValueError(f"No patient record found for case {case_id}")

        first_col = retrieved_df.columns[0]
        df_profile = pd.concat([retrieved_df[[first_col]], retrieved_df.loc[:, "Age":]], axis=1)
        df_profile_str = df_profile.to_csv(index=False)
        df_image_csv = retrieved_df.drop(columns=df_profile.columns).to_csv(index=False)

        print("Retrieved patient records: ", df_profile_str)
        state.add_raw("\nRetrieved patient records:" + df_profile_str)

        return retrieved_df, df_profile_str, df_image_csv

    def _turn_summarize_record(self, state: DialogueState, df_profile_str: str) -> str:
        summarize_prompt = self.fill_sum_p(SUMMARIZE_DATA, df_profile_str)
        summary = dp_answer(summarize_prompt)
        print("Agent: your records summary -- ", summary)
        state.add_agent("your records summary -- " + summary)
        state.summary = summary
        return summary

    def _turn_image_analysis(self, state: DialogueState, retrieved_df: pd.DataFrame, df_image_csv: str) -> str:
        print(state.round_three)
        state.add_raw(state.round_three)

        print("Agent: Next, I will analyze your meibomian gland images")
        state.add_agent("Next, I will analyze your meibomian gland images")

        print("Data obtained through image analysis: ", df_image_csv)
        state.add_raw("Data obtained through analysis: " + df_image_csv)

        d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9 = self.get_row_vars_by_pos(retrieved_df)
        image_prompt = self.fill_analyze_p(ANALYZE_DATA, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9)

        if d_1 == "Poor":
            print(IRREGULAR_RESPON)
            image_response = IRREGULAR_RESPON
        else:
            image_response = dp_answer(image_prompt)
            print("Agent: ", image_response)

        state.add_agent(image_response)
        state.image_response = image_response
        return image_response

    def _turn_rag_and_suggestion(
        self,
        state: DialogueState,
        symptom: str,
        detail: str,
        summary: str,
        df_image_csv: str,
        image_response: str,
    ) -> float:
        print("Agent: Let me check evidence-based treatment recommendations for similar cases")
        state.add_agent("Let me check evidence-based treatment recommendations for similar cases")

        query_text = symptom + detail + summary + df_image_csv + "Agent: " + image_response

        start_rag = time.perf_counter()
        top_k_chunks, _ = get_top_k_similar_chunks(query_text, self.all_text, self.model, k=3)
        end_rag = time.perf_counter()

        rag_elapsed = end_rag - start_rag
        print(f"rag耗时: {rag_elapsed:.5f} 秒")

        state.rag_info = ",".join(top_k_chunks)
        final_prompt = self.fill_last_p(LAST_ROUND, state.render(), state.rag_info)
        state.suggestions = dp_answer(final_prompt)

        print("Agent: Here are the suggestions -- ", state.suggestions)
        state.add_agent("Here are the suggestions -- " + state.suggestions)
        return rag_elapsed

    def run_one_case(self, case_index: int) -> Tuple[str, str, str]:
        start_all = time.perf_counter()

        state = DialogueState(
            case_index=case_index,
            symptom=self.round_columns.loc[case_index, "Round 1"],
            detail=self.round_columns.loc[case_index, "Round 2"],
            round_three=self.round_columns.loc[case_index, "Round 3"],
        )

        first_response = self._turn_opening(state)
        retrieved_df, df_profile_str, df_image_csv = self._turn_retrieve_record(state, first_response)
        summary = self._turn_summarize_record(state, df_profile_str)
        image_response = self._turn_image_analysis(state, retrieved_df, df_image_csv)
        rag_elapsed = self._turn_rag_and_suggestion(
            state,
            state.symptom,
            state.detail,
            summary,
            df_image_csv,
            image_response,
        )

        end_all = time.perf_counter()
        total_elapsed = end_all - start_all
        print(f"总耗时: {total_elapsed:.5f} 秒")

        return state.render(), f"{total_elapsed:.5f}s", f"{rag_elapsed:.5f}s"

    def run_all_cases(self) -> Tuple[List[str], List[str], List[str]]:
        conv_list: List[str] = []
        time_list: List[str] = []
        rag_time_list: List[str] = []

        for i in range(len(self.df_total)):
            conversation, total_time, rag_time = self.run_one_case(i)
            conv_list.append(conversation)
            time_list.append(total_time)
            rag_time_list.append(rag_time)

        print("总时间", time_list)
        print("rag时间", rag_time_list)

        return conv_list, time_list, rag_time_list
