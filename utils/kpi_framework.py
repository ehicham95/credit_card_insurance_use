import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class KPIFramework:
    """Success metrics and KPIs definition for cluster strategies"""
    
    def __init__(self, data, cluster_labels):
        self.data = data.copy()
        self.data['cluster'] = cluster_labels
        self.kpi_definitions = self._define_kpis()
        
    def _define_kpis(self):
        """Define comprehensive KPI framework"""
        return {
            'revenue_metrics': {
                'revenue_per_customer': {
                    'formula': 'total_revenue / customer_count',
                    'target_improvement': 0.1,
                    'frequency': 'monthly',
                    'priority': 'high'
                },
                'interchange_revenue': {
                    'formula': 'purchases * interchange_rate',
                    'target_improvement': 0.15,
                    'frequency': 'monthly',
                    'priority': 'high'
                },
                'interest_revenue': {
                    'formula': 'avg_balance * interest_rate',
                    'target_improvement': 0.08,
                    'frequency': 'monthly',
                    'priority': 'medium'
                }
            },
            'engagement_metrics': {
                'purchase_frequency': {
                    'formula': 'purchases_trx / active_months',
                    'target_improvement': 0.2,
                    'frequency': 'monthly',
                    'priority': 'high'
                },
                'utilization_rate': {
                    'formula': 'balance / credit_limit',
                    'target_improvement': 0.05,
                    'frequency': 'monthly',
                    'priority': 'medium'
                },
                'card_activation_rate': {
                    'formula': 'active_customers / total_customers',
                    'target_improvement': 0.1,
                    'frequency': 'quarterly',
                    'priority': 'high'
                }
            },
            'risk_metrics': {
                'default_rate': {
                    'formula': 'defaulted_customers / total_customers',
                    'target_improvement': -0.02,
                    'frequency': 'monthly',
                    'priority': 'high'
                },
                'payment_delinquency': {
                    'formula': 'late_payments / total_payments',
                    'target_improvement': -0.05,
                    'frequency': 'monthly',
                    'priority': 'high'
                },
                'credit_loss_rate': {
                    'formula': 'charge_offs / total_exposure',
                    'target_improvement': -0.01,
                    'frequency': 'quarterly',
                    'priority': 'high'
                }
            },
            'retention_metrics': {
                'customer_retention': {
                    'formula': 'retained_customers / previous_period_customers',
                    'target_improvement': 0.05,
                    'frequency': 'quarterly',
                    'priority': 'high'
                },
                'churn_rate': {
                    'formula': 'churned_customers / total_customers',
                    'target_improvement': -0.03,
                    'frequency': 'monthly',
                    'priority': 'high'
                }
            },
            'profitability_metrics': {
                'profit_per_customer': {
                    'formula': 'total_profit / customer_count',
                    'target_improvement': 0.12,
                    'frequency': 'monthly',
                    'priority': 'high'
                },
                'cost_to_serve': {
                    'formula': 'operational_costs / customer_count',
                    'target_improvement': -0.08,
                    'frequency': 'quarterly',
                    'priority': 'medium'
                }
            }
        }
    
    def calculate_baseline_kpis(self, cluster_id=None):
        """Calculate baseline KPIs for comparison"""
        
        if cluster_id is not None:
            data_subset = self.data[self.data['cluster'] == cluster_id]
        else:
            data_subset = self.data
        
        baseline_kpis = {}
        
        # Revenue metrics
        baseline_kpis['revenue_per_customer'] = (
            data_subset['PURCHASES'].mean() + data_subset['CASH_ADVANCE'].mean()
        )
        baseline_kpis['interchange_revenue'] = data_subset['PURCHASES'].mean() * 0.02
        baseline_kpis['interest_revenue'] = data_subset['BALANCE'].mean() * 0.18 / 12
        
        # Engagement metrics
        baseline_kpis['purchase_frequency'] = data_subset['PURCHASES_FREQUENCY'].mean()
        baseline_kpis['utilization_rate'] = data_subset['effective_utilization'].mean()
        baseline_kpis['card_activation_rate'] = (data_subset['PURCHASES'] > 0).mean()
        
        # Risk metrics (estimated)
        baseline_kpis['default_rate'] = self._estimate_default_rate(data_subset)
        baseline_kpis['payment_delinquency'] = (1 - data_subset['PRC_FULL_PAYMENT']).mean()
        
        # Retention metrics (estimated)
        baseline_kpis['customer_retention'] = 0.85  # Industry average
        baseline_kpis['churn_rate'] = 0.15
        
        # Profitability metrics
        revenue = baseline_kpis['revenue_per_customer']
        costs = data_subset['BALANCE'].mean() * 0.03 + 50  # Cost of funds + operational
        baseline_kpis['profit_per_customer'] = revenue * 0.2 - costs * 0.1
        baseline_kpis['cost_to_serve'] = costs
        
        return baseline_kpis
    
    def _estimate_default_rate(self, data_subset):
        """Estimate default rate based on risk indicators"""
        high_util = (data_subset['effective_utilization'] > 0.9).mean()
        min_pay_only = (data_subset['PRC_FULL_PAYMENT'] < 0.1).mean()
        return min(0.05 + high_util * 0.1 + min_pay_only * 0.05, 0.25)
    
    def create_kpi_scorecard(self, cluster_strategies):
        """Create KPI scorecard with targets"""
        
        scorecard_data = []
        
        for cluster_id, strategies in cluster_strategies.items():
            baseline = self.calculate_baseline_kpis(cluster_id)
            
            for strategy, expected_lift in strategies.items():
                for category, metrics in self.kpi_definitions.items():
                    for metric_name, metric_def in metrics.items():
                        
                        current_value = baseline.get(metric_name, 0)
                        target_lift = metric_def['target_improvement']
                        expected_value = current_value * (1 + expected_lift.get(metric_name, 0))
                        target_value = current_value * (1 + target_lift)
                        
                        scorecard_data.append({
                            'cluster': cluster_id,
                            'strategy': strategy,
                            'category': category,
                            'metric': metric_name,
                            'current_value': current_value,
                            'expected_value': expected_value,
                            'target_value': target_value,
                            'priority': metric_def['priority'],
                            'frequency': metric_def['frequency']
                        })
        
        return pd.DataFrame(scorecard_data)
    
    def monitoring_dashboard(self, cluster_id, time_period='monthly'):
        """Create monitoring dashboard template"""
        
        kpis = self.calculate_baseline_kpis(cluster_id)
        
        dashboard = {
            'cluster_id': cluster_id,
            'monitoring_period': time_period,
            'key_metrics': {},
            'alerts': [],
            'trends': {}
        }
        
        # High priority metrics for dashboard
        high_priority = [
            'revenue_per_customer', 'purchase_frequency', 'default_rate', 
            'customer_retention', 'profit_per_customer'
        ]
        
        for metric in high_priority:
            if metric in kpis:
                dashboard['key_metrics'][metric] = {
                    'current': kpis[metric],
                    'target': kpis[metric] * 1.1,  # 10% improvement target
                    'status': 'baseline'
                }
        
        return dashboard
    
    def success_criteria(self, strategy_type):
        """Define success criteria by strategy type"""
        
        criteria = {
            'retention_program': {
                'primary': 'customer_retention > 0.90',
                'secondary': ['churn_rate < 0.10', 'revenue_per_customer +15%'],
                'timeline': '6 months'
            },
            'activation_campaign': {
                'primary': 'card_activation_rate > 0.80',
                'secondary': ['purchase_frequency +25%', 'utilization_rate +10%'],
                'timeline': '3 months'
            },
            'risk_mitigation': {
                'primary': 'default_rate < 0.03',
                'secondary': ['payment_delinquency -20%', 'credit_loss_rate -15%'],
                'timeline': '12 months'
            },
            'revenue_optimization': {
                'primary': 'revenue_per_customer +20%',
                'secondary': ['interchange_revenue +25%', 'profit_per_customer +15%'],
                'timeline': '6 months'
            }
        }
        
        return criteria.get(strategy_type, {})
    
    def performance_tracking(self, cluster_id, actual_values, baseline_values):
        """Track performance against baselines"""
        
        performance = {}
        
        for metric, actual in actual_values.items():
            if metric in baseline_values:
                baseline = baseline_values[metric]
                improvement = (actual - baseline) / baseline
                
                # Get target improvement
                target = 0.1  # Default 10%
                for category in self.kpi_definitions.values():
                    if metric in category:
                        target = category[metric]['target_improvement']
                        break
                
                performance[metric] = {
                    'actual': actual,
                    'baseline': baseline,
                    'improvement': improvement,
                    'target': target,
                    'achieved_target': improvement >= target,
                    'performance_ratio': improvement / target if target != 0 else 0
                }
        
        # Overall performance score
        achieved_targets = sum(1 for p in performance.values() if p['achieved_target'])
        total_targets = len(performance)
        performance['overall_score'] = achieved_targets / total_targets if total_targets > 0 else 0
        
        return performance


## Usage Example:    
# kpi_framework = KPIFramework(df_risk_final, cluster_labels)
# baseline = kpi_framework.calculate_baseline_kpis(cluster_id=0)
# dashboard = kpi_framework.monitoring_dashboard(cluster_id=0)