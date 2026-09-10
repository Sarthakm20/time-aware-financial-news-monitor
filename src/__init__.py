from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
    DEFAULT_FINETUNED_MODEL_DIR,
    HALF_LIFE_DAYS,
    MONITORING_WINDOWS,
)

from src.preprocessing import (
    clean_phrasebank_dataframe,
    split_data_stratified,
    dataframe_to_hf_dataset,
    tokenize_hf_dataset,
)

from src.data_loading import (
    load_phrasebank_from_hub,
    load_split_csvs,
    stream_fnspid_slice,
)

from src.evaluate import (
    calculate_metrics,
    compute_confusion_matrix,
    export_error_analysis,
    save_metrics_json,
)

from src.inference import (
    load_fine_tuned_model,
    predict_sentiment_batch,
    score_news_dataframe,
)

from src.aggregation import (
    compute_recency_weights,
    summarize_sentiment_windows,
    compute_rolling_sentiment,
)

from src.visualization import (
    plot_confusion_matrix,
    plot_baseline_vs_finetuned,
    plot_rolling_sentiment_timeline,
)

__version__ = "0.1.0"

