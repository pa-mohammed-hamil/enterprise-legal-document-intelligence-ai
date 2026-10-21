"""
Predictive Analytics Engine

Advanced predictive analytics for contract outcomes, risk forecasting,
and performance prediction.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class ContractOutcomePrediction:
    """
    Predict contract outcomes based on historical data and contract features.
    
    Predictions:
    - Contract success probability
    - Likelihood of disputes
    - Renewal probability
    - Financial performance
    - Compliance risk
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize predictive models."""
        self.success_model = None
        self.dispute_model = None
        self.renewal_model = None
        self.financial_model = None
        self.scaler = StandardScaler()
        
        if model_path:
            self.load_models(model_path)

    def predict_contract_success(self, contract_features: Dict) -> Dict:
        """
        Predict probability of contract success.
        
        Args:
            contract_features: Dictionary of contract attributes
            
        Returns:
            Dictionary with success probability and contributing factors
        """
        features = self._extract_features(contract_features)
        
        if self.success_model is None:
            # Use rule-based prediction if model not trained
            return self._rule_based_success_prediction(contract_features)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        success_prob = self.success_model.predict_proba(features_scaled)[0][1]
        
        # Get feature importance
        feature_importance = dict(zip(
            self._get_feature_names(),
            self.success_model.feature_importances_
        ))
        
        # Sort by importance
        top_factors = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "success_probability": float(success_prob),
            "risk_level": self._categorize_risk(success_prob),
            "confidence": self._calculate_confidence(features_scaled),
            "top_factors": [
                {"factor": name, "importance": float(imp)}
                for name, imp in top_factors
            ],
            "prediction_date": datetime.utcnow().isoformat(),
        }

    def predict_dispute_likelihood(self, contract_features: Dict) -> Dict:
        """
        Predict likelihood of contract disputes.
        
        Args:
            contract_features: Dictionary of contract attributes
            
        Returns:
            Dictionary with dispute probability and risk factors
        """
        features = self._extract_features(contract_features)
        
        # Dispute risk factors
        dispute_indicators = {
            "ambiguous_clauses": contract_features.get("ambiguous_clause_count", 0),
            "missing_clauses": contract_features.get("missing_key_clauses", []),
            "complexity_score": contract_features.get("complexity_score", 0),
            "party_history": contract_features.get("party_dispute_history", 0),
            "contract_value": contract_features.get("contract_value", 0),
            "duration_months": contract_features.get("duration_months", 0),
        }
        
        # Calculate dispute score (0-100)
        dispute_score = 0
        
        # Ambiguous clauses increase risk
        dispute_score += min(dispute_indicators["ambiguous_clauses"] * 5, 30)
        
        # Missing key clauses increase risk
        dispute_score += len(dispute_indicators["missing_clauses"]) * 8
        
        # High complexity increases risk
        if dispute_indicators["complexity_score"] > 0.7:
            dispute_score += 20
        
        # Party history
        dispute_score += min(dispute_indicators["party_history"] * 10, 25)
        
        # High value contracts have higher stakes
        if dispute_indicators["contract_value"] > 1000000:
            dispute_score += 10
        
        # Cap at 100
        dispute_score = min(dispute_score, 100)
        dispute_probability = dispute_score / 100
        
        return {
            "dispute_probability": float(dispute_probability),
            "dispute_score": int(dispute_score),
            "risk_level": self._categorize_dispute_risk(dispute_score),
            "risk_factors": self._identify_dispute_factors(dispute_indicators),
            "mitigation_suggestions": self._suggest_dispute_mitigations(dispute_indicators),
            "estimated_cost_range": self._estimate_dispute_cost(
                dispute_probability,
                dispute_indicators["contract_value"]
            ),
        }

    def predict_renewal_probability(self, contract_features: Dict) -> Dict:
        """
        Predict probability of contract renewal.
        
        Args:
            contract_features: Dictionary of contract attributes
            
        Returns:
            Dictionary with renewal probability and influencing factors
        """
        renewal_factors = {
            "performance_score": contract_features.get("performance_score", 0.5),
            "satisfaction_score": contract_features.get("satisfaction_score", 0.5),
            "issue_count": contract_features.get("issue_count", 0),
            "payment_history": contract_features.get("payment_punctuality", 1.0),
            "relationship_duration": contract_features.get("relationship_months", 0),
            "market_conditions": contract_features.get("market_favorability", 0.5),
            "competitive_alternatives": contract_features.get("alternatives_count", 0),
        }
        
        # Calculate renewal probability
        renewal_score = 0.3  # Base probability
        
        # Strong performance increases probability
        renewal_score += renewal_factors["performance_score"] * 0.25
        
        # Customer satisfaction
        renewal_score += renewal_factors["satisfaction_score"] * 0.2
        
        # Issues decrease probability
        renewal_score -= min(renewal_factors["issue_count"] * 0.05, 0.2)
        
        # Payment history
        renewal_score += renewal_factors["payment_history"] * 0.15
        
        # Long relationship increases probability
        if renewal_factors["relationship_duration"] > 24:
            renewal_score += 0.1
        
        # Market conditions
        renewal_score += (renewal_factors["market_conditions"] - 0.5) * 0.1
        
        # Competition decreases probability
        renewal_score -= min(renewal_factors["competitive_alternatives"] * 0.03, 0.15)
        
        # Clamp between 0 and 1
        renewal_probability = max(0, min(1, renewal_score))
        
        return {
            "renewal_probability": float(renewal_probability),
            "confidence_level": self._calculate_renewal_confidence(renewal_factors),
            "key_factors": self._identify_renewal_factors(renewal_factors),
            "churn_risk": 1 - renewal_probability,
            "recommended_actions": self._suggest_renewal_actions(
                renewal_probability,
                renewal_factors
            ),
            "optimal_renewal_timing": self._calculate_optimal_renewal_timing(
                contract_features
            ),
        }

    def predict_financial_performance(self, contract_features: Dict) -> Dict:
        """
        Predict financial performance of a contract.
        
        Args:
            contract_features: Dictionary of contract attributes
            
        Returns:
            Dictionary with financial predictions
        """
        contract_value = contract_features.get("contract_value", 0)
        duration_months = contract_features.get("duration_months", 12)
        payment_terms = contract_features.get("payment_terms", "net_30")
        
        # Calculate metrics
        monthly_revenue = contract_value / duration_months if duration_months > 0 else 0
        
        # Payment delay risk
        payment_delay_prob = self._calculate_payment_delay_risk(
            contract_features.get("client_payment_history", 1.0),
            payment_terms
        )
        
        # Revenue recognition pattern
        revenue_pattern = self._predict_revenue_pattern(
            contract_value,
            duration_months,
            payment_terms
        )
        
        # Cost estimation
        estimated_costs = self._estimate_contract_costs(contract_features)
        
        # Profitability
        estimated_profit = contract_value - estimated_costs
        profit_margin = (estimated_profit / contract_value * 100) if contract_value > 0 else 0
        
        return {
            "contract_value": float(contract_value),
            "estimated_revenue": float(contract_value),
            "estimated_costs": float(estimated_costs),
            "estimated_profit": float(estimated_profit),
            "profit_margin_percent": float(profit_margin),
            "monthly_revenue": float(monthly_revenue),
            "payment_delay_probability": float(payment_delay_prob),
            "expected_payment_delay_days": self._estimate_payment_delay(payment_delay_prob),
            "revenue_pattern": revenue_pattern,
            "roi_estimate": self._calculate_roi(contract_features),
            "financial_risk_score": self._calculate_financial_risk(
                profit_margin,
                payment_delay_prob
            ),
        }

    def predict_compliance_risk(self, contract_features: Dict) -> Dict:
        """
        Predict compliance and regulatory risk.
        
        Args:
            contract_features: Dictionary of contract attributes
            
        Returns:
            Dictionary with compliance risk assessment
        """
        compliance_factors = {
            "jurisdiction": contract_features.get("jurisdiction", "unknown"),
            "regulatory_complexity": contract_features.get("regulatory_complexity", 0.5),
            "data_handling": contract_features.get("handles_sensitive_data", False),
            "international": contract_features.get("cross_border", False),
            "industry": contract_features.get("industry", "general"),
            "audit_trail": contract_features.get("has_audit_trail", True),
            "compliance_clauses": contract_features.get("compliance_clause_count", 0),
        }
        
        # Calculate compliance risk
        risk_score = 0
        
        # Regulatory complexity
        risk_score += compliance_factors["regulatory_complexity"] * 30
        
        # Sensitive data handling
        if compliance_factors["data_handling"]:
            risk_score += 20
        
        # International contracts have higher risk
        if compliance_factors["international"]:
            risk_score += 15
        
        # Missing audit trail
        if not compliance_factors["audit_trail"]:
            risk_score += 15
        
        # Few compliance clauses
        if compliance_factors["compliance_clauses"] < 3:
            risk_score += 10
        
        # Industry-specific risk
        high_risk_industries = ["healthcare", "finance", "defense"]
        if compliance_factors["industry"] in high_risk_industries:
            risk_score += 10
        
        return {
            "compliance_risk_score": float(risk_score),
            "risk_level": self._categorize_compliance_risk(risk_score),
            "regulatory_requirements": self._identify_regulatory_requirements(
                compliance_factors
            ),
            "compliance_gaps": self._identify_compliance_gaps(compliance_factors),
            "recommended_clauses": self._recommend_compliance_clauses(compliance_factors),
            "audit_frequency": self._recommend_audit_frequency(risk_score),
        }

    def generate_outcome_forecast(
        self,
        contract_features: Dict,
        forecast_months: int = 12
    ) -> Dict:
        """
        Generate comprehensive forecast for contract outcomes.
        
        Args:
            contract_features: Dictionary of contract attributes
            forecast_months: Number of months to forecast
            
        Returns:
            Comprehensive forecast with multiple dimensions
        """
        # Get all predictions
        success_pred = self.predict_contract_success(contract_features)
        dispute_pred = self.predict_dispute_likelihood(contract_features)
        renewal_pred = self.predict_renewal_probability(contract_features)
        financial_pred = self.predict_financial_performance(contract_features)
        compliance_pred = self.predict_compliance_risk(contract_features)
        
        # Generate timeline forecast
        timeline_forecast = self._generate_timeline_forecast(
            contract_features,
            forecast_months
        )
        
        # Overall risk assessment
        overall_risk = self._calculate_overall_risk(
            success_pred["success_probability"],
            dispute_pred["dispute_probability"],
            compliance_pred["compliance_risk_score"] / 100
        )
        
        return {
            "contract_id": contract_features.get("contract_id"),
            "forecast_date": datetime.utcnow().isoformat(),
            "forecast_period_months": forecast_months,
            "overall_risk_score": float(overall_risk),
            "overall_risk_level": self._categorize_risk(1 - overall_risk),
            "success_prediction": success_pred,
            "dispute_prediction": dispute_pred,
            "renewal_prediction": renewal_pred,
            "financial_prediction": financial_pred,
            "compliance_prediction": compliance_pred,
            "timeline_forecast": timeline_forecast,
            "key_recommendations": self._generate_recommendations(
                success_pred,
                dispute_pred,
                renewal_pred,
                financial_pred,
                compliance_pred
            ),
        }

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _extract_features(self, contract_features: Dict) -> List[float]:
        """Extract numerical features for ML models."""
        return [
            contract_features.get("contract_value", 0),
            contract_features.get("duration_months", 12),
            contract_features.get("complexity_score", 0.5),
            contract_features.get("party_risk_score", 0.5),
            contract_features.get("clause_count", 0),
            contract_features.get("amendment_count", 0),
            int(contract_features.get("has_indemnity", False)),
            int(contract_features.get("has_termination", True)),
            contract_features.get("negotiation_rounds", 1),
        ]

    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        return [
            "contract_value",
            "duration_months",
            "complexity_score",
            "party_risk_score",
            "clause_count",
            "amendment_count",
            "has_indemnity",
            "has_termination",
            "negotiation_rounds",
        ]

    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level based on probability."""
        if probability >= 0.8:
            return "very_low"
        elif probability >= 0.6:
            return "low"
        elif probability >= 0.4:
            return "medium"
        elif probability >= 0.2:
            return "high"
        else:
            return "very_high"

    def _categorize_dispute_risk(self, score: int) -> str:
        """Categorize dispute risk level."""
        if score < 20:
            return "low"
        elif score < 40:
            return "medium"
        elif score < 70:
            return "high"
        else:
            return "critical"

    def _categorize_compliance_risk(self, score: float) -> str:
        """Categorize compliance risk level."""
        if score < 25:
            return "low"
        elif score < 50:
            return "medium"
        elif score < 75:
            return "high"
        else:
            return "critical"

    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate prediction confidence."""
        # Placeholder: implement based on model uncertainty
        return 0.85

    def _rule_based_success_prediction(self, features: Dict) -> Dict:
        """Fallback rule-based prediction when model unavailable."""
        # Simple heuristic
        score = 0.5
        
        if features.get("has_indemnity", False):
            score += 0.1
        if features.get("has_termination", True):
            score += 0.1
        if features.get("complexity_score", 0.5) < 0.5:
            score += 0.15
        if features.get("party_risk_score", 0.5) < 0.3:
            score += 0.15
        
        return {
            "success_probability": min(score, 1.0),
            "risk_level": self._categorize_risk(score),
            "confidence": 0.6,
            "top_factors": [],
            "prediction_date": datetime.utcnow().isoformat(),
        }

    def _identify_dispute_factors(self, indicators: Dict) -> List[Dict]:
        """Identify key dispute risk factors."""
        factors = []
        
        if indicators["ambiguous_clauses"] > 3:
            factors.append({
                "factor": "Ambiguous clauses",
                "severity": "high",
                "count": indicators["ambiguous_clauses"]
            })
        
        if indicators["missing_clauses"]:
            factors.append({
                "factor": "Missing key clauses",
                "severity": "medium",
                "clauses": indicators["missing_clauses"]
            })
        
        if indicators["complexity_score"] > 0.7:
            factors.append({
                "factor": "High complexity",
                "severity": "medium",
                "score": indicators["complexity_score"]
            })
        
        return factors

    def _suggest_dispute_mitigations(self, indicators: Dict) -> List[str]:
        """Suggest actions to mitigate dispute risk."""
        suggestions = []
        
        if indicators["ambiguous_clauses"] > 0:
            suggestions.append("Clarify ambiguous clauses with explicit definitions")
        
        if indicators["missing_clauses"]:
            suggestions.append(f"Add missing clauses: {', '.join(indicators['missing_clauses'])}")
        
        if indicators["complexity_score"] > 0.7:
            suggestions.append("Simplify contract language and structure")
        
        suggestions.append("Include detailed dispute resolution process")
        suggestions.append("Add mediation clause before arbitration")
        
        return suggestions

    def _estimate_dispute_cost(self, probability: float, contract_value: float) -> Dict:
        """Estimate potential dispute costs."""
        # Typical dispute costs range from 10-30% of contract value
        min_cost = contract_value * 0.10 * probability
        max_cost = contract_value * 0.30 * probability
        
        return {
            "min_cost": float(min_cost),
            "max_cost": float(max_cost),
            "expected_cost": float((min_cost + max_cost) / 2)
        }

    def _calculate_renewal_confidence(self, factors: Dict) -> str:
        """Calculate confidence in renewal prediction."""
        # Based on data quality and historical patterns
        if factors["relationship_duration"] > 24:
            return "high"
        elif factors["relationship_duration"] > 12:
            return "medium"
        else:
            return "low"

    def _identify_renewal_factors(self, factors: Dict) -> List[Dict]:
        """Identify key factors affecting renewal."""
        key_factors = []
        
        if factors["performance_score"] > 0.7:
            key_factors.append({
                "factor": "Strong performance",
                "impact": "positive",
                "score": factors["performance_score"]
            })
        
        if factors["issue_count"] > 5:
            key_factors.append({
                "factor": "Multiple issues",
                "impact": "negative",
                "count": factors["issue_count"]
            })
        
        return key_factors

    def _suggest_renewal_actions(
        self,
        probability: float,
        factors: Dict
    ) -> List[str]:
        """Suggest actions to improve renewal probability."""
        actions = []
        
        if probability < 0.5:
            actions.append("Schedule urgent client satisfaction review")
            actions.append("Address outstanding issues immediately")
        
        if factors["performance_score"] < 0.6:
            actions.append("Develop performance improvement plan")
        
        if factors["issue_count"] > 3:
            actions.append("Prioritize issue resolution")
        
        actions.append("Initiate renewal discussions 90 days before expiry")
        
        return actions

    def _calculate_optimal_renewal_timing(self, features: Dict) -> Dict:
        """Calculate optimal timing for renewal discussions."""
        expiry_date = features.get("expiry_date")
        if not expiry_date:
            return {}
        
        # Start discussions 90-120 days before expiry
        optimal_start = expiry_date - timedelta(days=120)
        optimal_end = expiry_date - timedelta(days=90)
        
        return {
            "optimal_start_date": optimal_start.isoformat(),
            "optimal_end_date": optimal_end.isoformat(),
            "days_until_optimal_start": (optimal_start - datetime.utcnow()).days
        }

    def _calculate_payment_delay_risk(
        self,
        payment_history: float,
        payment_terms: str
    ) -> float:
        """Calculate probability of payment delays."""
        base_risk = 1 - payment_history
        
        # Payment terms affect risk
        term_multiplier = {
            "immediate": 0.5,
            "net_30": 1.0,
            "net_60": 1.3,
            "net_90": 1.5,
        }.get(payment_terms, 1.0)
        
        return min(base_risk * term_multiplier, 1.0)

    def _predict_revenue_pattern(
        self,
        total_value: float,
        duration: int,
        payment_terms: str
    ) -> List[Dict]:
        """Predict revenue recognition pattern over time."""
        monthly_revenue = total_value / duration if duration > 0 else 0
        
        pattern = []
        for month in range(duration):
            pattern.append({
                "month": month + 1,
                "expected_revenue": float(monthly_revenue),
                "cumulative_revenue": float(monthly_revenue * (month + 1))
            })
        
        return pattern

    def _estimate_contract_costs(self, features: Dict) -> float:
        """Estimate costs associated with contract execution."""
        # Placeholder: implement detailed cost model
        contract_value = features.get("contract_value", 0)
        complexity = features.get("complexity_score", 0.5)
        
        # Typical costs: 20-40% of revenue
        base_cost_ratio = 0.30
        complexity_adjustment = complexity * 0.1
        
        return contract_value * (base_cost_ratio + complexity_adjustment)

    def _estimate_payment_delay(self, delay_prob: float) -> int:
        """Estimate expected payment delay in days."""
        # Higher probability = longer delay
        return int(delay_prob * 30)  # Max 30 days delay

    def _calculate_roi(self, features: Dict) -> float:
        """Calculate estimated return on investment."""
        revenue = features.get("contract_value", 0)
        costs = self._estimate_contract_costs(features)
        investment = costs * 0.5  # Assume 50% upfront
        
        if investment == 0:
            return 0
        
        roi = ((revenue - costs) / investment) * 100
        return float(roi)

    def _calculate_financial_risk(
        self,
        profit_margin: float,
        payment_delay_prob: float
    ) -> float:
        """Calculate overall financial risk score."""
        risk = 0
        
        # Low margin increases risk
        if profit_margin < 10:
            risk += 40
        elif profit_margin < 20:
            risk += 20
        
        # Payment delays increase risk
        risk += payment_delay_prob * 60
        
        return min(risk, 100)

    def _identify_regulatory_requirements(self, factors: Dict) -> List[str]:
        """Identify applicable regulatory requirements."""
        requirements = []
        
        if factors["data_handling"]:
            requirements.append("GDPR compliance required")
            requirements.append("Data protection impact assessment")
        
        if factors["international"]:
            requirements.append("Cross-border data transfer agreements")
        
        if factors["industry"] == "healthcare":
            requirements.append("HIPAA compliance")
        elif factors["industry"] == "finance":
            requirements.append("SOX compliance")
            requirements.append("Anti-money laundering checks")
        
        return requirements

    def _identify_compliance_gaps(self, factors: Dict) -> List[str]:
        """Identify compliance gaps."""
        gaps = []
        
        if not factors["audit_trail"]:
            gaps.append("Missing audit trail mechanism")
        
        if factors["compliance_clauses"] < 3:
            gaps.append("Insufficient compliance clauses")
        
        if factors["data_handling"] and factors["compliance_clauses"] == 0:
            gaps.append("No data protection clauses")
        
        return gaps

    def _recommend_compliance_clauses(self, factors: Dict) -> List[str]:
        """Recommend compliance clauses to add."""
        clauses = []
        
        if factors["data_handling"]:
            clauses.append("Data processing agreement")
            clauses.append("Data breach notification clause")
        
        if factors["international"]:
            clauses.append("Governing law and jurisdiction clause")
        
        clauses.append("Compliance with laws clause")
        clauses.append("Audit rights clause")
        
        return clauses

    def _recommend_audit_frequency(self, risk_score: float) -> str:
        """Recommend audit frequency based on risk."""
        if risk_score > 75:
            return "quarterly"
        elif risk_score > 50:
            return "semi_annual"
        else:
            return "annual"

    def _generate_timeline_forecast(
        self,
        features: Dict,
        months: int
    ) -> List[Dict]:
        """Generate month-by-month forecast."""
        timeline = []
        
        for month in range(1, months + 1):
            # Forecast changes over time
            timeline.append({
                "month": month,
                "success_probability": 0.8 - (month * 0.01),  # Slight degradation
                "dispute_risk": 0.1 + (month * 0.005),
                "financial_health": 0.9 - (month * 0.01),
            })
        
        return timeline

    def _calculate_overall_risk(
        self,
        success_prob: float,
        dispute_prob: float,
        compliance_risk: float
    ) -> float:
        """Calculate overall risk score."""
        # Weighted average
        overall = (
            success_prob * 0.4 +
            (1 - dispute_prob) * 0.3 +
            (1 - compliance_risk) * 0.3
        )
        return float(overall)

    def _generate_recommendations(
        self,
        success: Dict,
        dispute: Dict,
        renewal: Dict,
        financial: Dict,
        compliance: Dict
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if success["success_probability"] < 0.6:
            recommendations.append("Review and strengthen contract terms")
        
        if dispute["dispute_probability"] > 0.4:
            recommendations.extend(dispute["mitigation_suggestions"][:2])
        
        if renewal["renewal_probability"] < 0.5:
            recommendations.extend(renewal["recommended_actions"][:2])
        
        if financial["profit_margin_percent"] < 15:
            recommendations.append("Renegotiate pricing or reduce costs")
        
        if compliance["risk_level"] in ["high", "critical"]:
            recommendations.append("Urgent compliance review required")
        
        return recommendations[:5]  # Top 5 recommendations

    def load_models(self, model_path: str):
        """Load pre-trained models."""
        try:
            self.success_model = joblib.load(f"{model_path}/success_model.pkl")
            self.dispute_model = joblib.load(f"{model_path}/dispute_model.pkl")
            self.renewal_model = joblib.load(f"{model_path}/renewal_model.pkl")
            self.financial_model = joblib.load(f"{model_path}/financial_model.pkl")
            self.scaler = joblib.load(f"{model_path}/scaler.pkl")
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load models: {e}. Using rule-based predictions.")

    def save_models(self, model_path: str):
        """Save trained models."""
        joblib.dump(self.success_model, f"{model_path}/success_model.pkl")
        joblib.dump(self.dispute_model, f"{model_path}/dispute_model.pkl")
        joblib.dump(self.renewal_model, f"{model_path}/renewal_model.pkl")
        joblib.dump(self.financial_model, f"{model_path}/financial_model.pkl")
        joblib.dump(self.scaler, f"{model_path}/scaler.pkl")
        logger.info("Models saved successfully")
