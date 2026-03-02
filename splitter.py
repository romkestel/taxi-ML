import polars as pl
import os


def split_data(input_path, train_path, test_path, test_size=0.1):
    print(f"📂 Loading refined data from {input_path}...")

    # Load the whole dataset
    df = pl.read_parquet(input_path)

    # 1. Shuffle the data (Crucial so the model doesn't learn 'date order')
    # Ted would call this 'Ensuring IID' (Independent and Identically Distributed)
    df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)

    # 2. Calculate the split point
    total_rows = len(df_shuffled)
    test_count = int(total_rows * test_size)
    train_count = total_rows - test_count

    # 3. Slice the data
    train_df = df_shuffled.head(train_count)
    test_df = df_shuffled.tail(test_count)

    # 4. Save the splits
    train_df.write_parquet(train_path)
    test_df.write_parquet(test_path)

    print(f"✅ Split Complete!")
    print(f"Total Rows: {total_rows}")
    print(f"Training Set: {len(train_df)} rows ({train_path})")
    print(f"Test Set: {len(test_df)} rows ({test_path})")


if __name__ == "__main__":
    SOURCE = "./data/refined_whole_source.parquet"
    TRAIN_OUT = "./data/train_data.parquet"
    TEST_OUT = "./data/test_data.parquet"

    if os.path.exists(SOURCE):
        split_data(SOURCE, TRAIN_OUT, TEST_OUT)
    else:
        print(f"❌ Could not find {SOURCE}. Run refinery.py first!")
