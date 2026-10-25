"""State container with modules sharing the same cleaned monthly/annual tables."""
import pandas as pd
from .data import DataMixin
from .annual import AnnualMixin
from .monthly import MonthlyMixin
from .mining import MiningMixin
from .decisions import DecisionMixin

class CoconutWorkbench(DataMixin, AnnualMixin, MonthlyMixin, MiningMixin, DecisionMixin):
    def __init__(self):
        self.monthly = None
        self.annual = None
        self.audit = {}
        self.source = None
        self.source_name = ""
        self.reset_models()
        self.reset_mining()

    def reset_mining(self):
        self.monthly_comparison=pd.DataFrame();self.monthly_predictions=pd.DataFrame();self.monthly_models={}
        self.monthly_winner=None;self.weight_report=pd.DataFrame();self.training_weights=pd.DataFrame()
        self.classification_metrics=pd.DataFrame();self.classification_predictions=pd.DataFrame()
        self.class_counts=pd.DataFrame();self.class_details={};self.classifier=None
        self.clusters=pd.DataFrame();self.cluster_scores=pd.DataFrame();self.anomalies=pd.DataFrame()
        self.three_month_predictions=pd.DataFrame();self.mining_settings={};self.monthly_reliance=pd.DataFrame()
