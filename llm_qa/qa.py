import argparse

from data.data_loader import read_csv_file, save_list_to_txt
from llm_qa.conversation_engine import MultiTurnConversationEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_path", type=str, help="documents for RAG", default="data/txt_files")
    parser.add_argument("--data_path", type=str, help="patients records", default="data/dry_eye_data_v3.csv")
    args = parser.parse_args()

    df_total = read_csv_file(args.data_path)
    round_columns = df_total.iloc[:, -3:]
    patient_profile_df = df_total.drop(columns=round_columns.columns)

    print(patient_profile_df)

    engine = MultiTurnConversationEngine(
        df_total=patient_profile_df,
        round_columns=round_columns,
        doc_path=args.doc_path,
    )

    conv_list, time_list, rag_time_list = engine.run_all_cases()

    save_list_to_txt(conv_list, "results/conversations_all.txt")
    save_list_to_txt(time_list, "results/total_time.txt")
    save_list_to_txt(rag_time_list, "results/rag_time.txt")


if __name__ == "__main__":
    main()
