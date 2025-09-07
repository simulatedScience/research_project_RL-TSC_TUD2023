"""
Author: Google LearnLM 2.0 Flash
"""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy.contrasts import Treatment  # For custom contrasts
import tkinter as tk
from tkinter import filedialog

# Load the data
root = tk.Tk()
root.withdraw()  # Hide the main window
file_path = filedialog.askopenfilename(title="Select the CSV data file", filetypes=[("CSV files", "*.csv")])
data = pd.read_csv(file_path)

# --- 1. Data Preprocessing and Exploration ---
# Convert categorical variables to appropriate types
data['exp_identifier'] = data['exp_identifier'].astype('category')
data['agent'] = data['agent'].astype('category')

# Explore data - check distributions, correlations, etc.
print(data.describe())
print(data[['fc', 'tpr', 'fpr', 'travel time', 'throughput']].corr())


# --- 2. Independent Models per Agent Type ---
# We build separate models for MaxPressure, Disturbed, and Undisturbed.
# Disturbed/Undisturbed have agent entries like "Disturbed, seed=134"; we
# normalize these to an agent_type column for filtering.
def infer_agent_type(agent_value: str) -> str:
    s = str(agent_value)
    if s.startswith('Disturbed'):
        return 'Disturbed'
    if s.startswith('Undisturbed'):
        return 'Undisturbed'
    if s.startswith('MaxPressure'):
        return 'MaxPressure'
    return s


data['agent_type'] = data['agent'].astype(str).apply(infer_agent_type).astype('category')


def run_models_for_agent_type(df: pd.DataFrame, agent_type: str) -> None:
    df_type = df[df['agent_type'] == agent_type].copy()

    if df_type.empty:
        print(f"\nNo data for agent type: {agent_type}")
        return

    # Re-center predictors within this agent type subset
    for col in ['fc', 'tpr', 'fpr']:
        df_type[f'{col}_c'] = df_type[col] - df_type[col].mean()

    # Formulas
    formula_tt = "Q('travel time') ~ fc_c * tpr_c + fc_c * fpr_c + tpr_c * fpr_c"
    formula_tp = "throughput ~ fc_c * tpr_c + fc_c * fpr_c + tpr_c * fpr_c"

    # Decide on MixedLM vs OLS depending on number of groups (seeds/agents) available
    n_groups = df_type['agent'].nunique()
    print(f"\n=== Agent Type: {agent_type} | n_rows={len(df_type)} | n_groups(agent)={n_groups} ===")

    # Travel time model
    if n_groups >= 2:
        try:
            model_tt = smf.mixedlm(formula_tt, df_type, re_formula="~1", groups=df_type['agent']).fit()
            print("\nMixedLM (Travel Time) Summary:")
            print(model_tt.summary())
        except Exception as e:
            print(f"MixedLM failed for Travel Time (falling back to OLS): {e}")
            model_tt = smf.ols(formula_tt, df_type).fit()
            print("\nOLS (Travel Time) Summary:")
            print(model_tt.summary())
    else:
        model_tt = smf.ols(formula_tt, df_type).fit()
        print("\nOLS (Travel Time) Summary:")
        print(model_tt.summary())

    # ANOVA for fixed effects (Type III)
    fe_tt = smf.ols(formula_tt, df_type).fit()
    anova_tt = sm.stats.anova_lm(fe_tt, typ=3)
    print("\nFixed Effects ANOVA (Travel Time, Type III):")
    print(anova_tt)

    # Throughput model
    if n_groups >= 2:
        try:
            model_tp = smf.mixedlm(formula_tp, df_type, re_formula="~1", groups=df_type['agent']).fit()
            print("\nMixedLM (Throughput) Summary:")
            print(model_tp.summary())
        except Exception as e:
            print(f"MixedLM failed for Throughput (falling back to OLS): {e}")
            model_tp = smf.ols(formula_tp, df_type).fit()
            print("\nOLS (Throughput) Summary:")
            print(model_tp.summary())
    else:
        model_tp = smf.ols(formula_tp, df_type).fit()
        print("\nOLS (Throughput) Summary:")
        print(model_tp.summary())

    fe_tp = smf.ols(formula_tp, df_type).fit()
    anova_tp = sm.stats.anova_lm(fe_tp, typ=3)
    print("\nFixed Effects ANOVA (Throughput, Type III):")
    print(anova_tp)


# Run per-agent-type analyses
for atype in ['MaxPressure', 'Disturbed', 'Undisturbed']:
    run_models_for_agent_type(data, atype)


# --- 4. Model Diagnostics ---
# For brevity, diagnostics plotting is omitted. Consider checking residuals,
# Q-Q plots, and heteroscedasticity for each fitted model above as needed.

# --- 5. Post-Hoc Tests and Contrasts (If Necessary) ---
# If you introduce categorical predictors with >2 levels in the future, you can
# define appropriate contrasts and run post-hoc comparisons within the agent
# type subsets using a similar pattern as above.