from .data_generator import generate_journeys, journey_summary, top_paths, CHANNELS, CHANNEL_LABELS
from .attribution import (
    last_touch, first_touch, linear_touch, time_decay, position_based,
    markov_chain, shapley_exact, shapley_ordered, banzhaf,
    shapley_interaction_index, run_all_models, shapley_bootstrap_ci,
)
from .optimizer import optimize_budget

try:
    from .charts import (
        attribution_comparison, shapley_waterfall, model_radar,
        interaction_heatmap, journey_sankey, budget_waterfall,
        budget_delta_chart, markov_transition_heatmap,
        channel_funnel_bar, conversion_rate_bar,
        shapley_ci_chart,
    )
except ImportError:
    pass  # plotly not installed; charts unavailable

from .mmm_data_generator import (
    generate_mmm_data,
    get_channel_contribution_shares,
    mmm_summary_stats,
    CHANNEL_TRUE_CONTRIBUTION,
    ADSTOCK_DECAY,
)

from .hybrid_attribution import (
    blend_mta_mmm,
    compute_unified_metrics,
    compare_mta_vs_mmm_vs_hybrid,
    offline_credit_recovery,
    DEFAULT_ALPHA,
)