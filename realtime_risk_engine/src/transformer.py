import pandas as pd
import numpy as np
from datetime import datetime

class UniversalTransformer:
    """
    Standardises transaction schemas into the 
    Universal Ledger format used by the VECTOR engine.
    """
    
    @staticmethod
    def get_tag(description: str, tx_type: str) -> str:
        """
        Maps transaction descriptions and types to universal tags.
        """
        desc = str(description).upper()
        ty = str(tx_type).upper()
        
        # Salary / Income mapping
        salary_keywords = ["SALARY", "WAGES", "PAYROLL", "BGC", "FPI"]
        if any(kw in desc for kw in salary_keywords) or ty in ["BGC", "FPI"]:
            return "SALARY"
        
        # Bills / Standing Orders mapping
        bill_keywords = ["BILL", "RENT", "UTILITY", "TAX", "INSURANCE", "DD", "SIPO"]
        if any(kw in desc for kw in bill_keywords) or ty == "DD":
            return "BILL"
            
        return "UNCATEGORIZED"

    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a raw transaction DataFrame to the Universal Ledger.
        Expects columns: ['Transaction Date', 'Transaction Description', 
                        'Transaction Type', 'Debit Amount', 'Credit Amount', 'Balance']
        """
        ledger = pd.DataFrame()
        
        # Mapping Dates with robust format handling
        ledger["date"] = pd.to_datetime(df["Transaction Date"], dayfirst=True, errors='coerce')
        
        # Mapping Cashflows
        ledger["cash_in"] = df["Credit Amount"].fillna(0.0).astype(float)
        ledger["cash_out"] = df["Debit Amount"].fillna(0.0).astype(float)
        ledger["balance"] = df["Balance"].astype(float)
        
        # Applying Universal Tagging
        ledger["tag"] = df.apply(
            lambda x: self.get_tag(x["Transaction Description"], x["Transaction Type"]), 
            axis=1
        )
        
        # Explicitly exclude internal loan payments if present to prevent leakage
        ledger = ledger[ledger["tag"] != "LOAN_PAYMENT"]
        
        return ledger.sort_values("date").reset_index(drop=True)
