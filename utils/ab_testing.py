import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class ABTestingFramework:
    """A/B testing framework for cluster-based strategies"""
    
    def __init__(self, data, cluster_labels):
        self.data = data.copy()
        self.data['cluster'] = cluster_labels
        self.experiments = {}
        
    def design_experiment(self, cluster_id, strategy_name, test_ratio=0.5, 
                        duration_days=90, min_sample_size=100):
        """Design A/B test for cluster strategy"""
        
        cluster_data = self.data[self.data['cluster'] == cluster_id]
        
        if len(cluster_data) < min_sample_size:
            raise ValueError(f"Insufficient sample size: {len(cluster_data)} < {min_sample_size}")
        
        # Random assignment to test/control
        np.random.seed(42)
        test_assignment = np.random.choice([0, 1], size=len(cluster_data), 
                                        p=[1-test_ratio, test_ratio])
        
        experiment_design = {
            'cluster_id': cluster_id,
            'strategy_name': strategy_name,
            'total_customers': len(cluster_data),
            'test_group_size': np.sum(test_assignment),
            'control_group_size': len(cluster_data) - np.sum(test_assignment),
            'test_ratio': test_ratio,
            'duration_days': duration_days,
            'start_date': datetime.now(),
            'end_date': datetime.now() + timedelta(days=duration_days),
            'customer_ids': cluster_data.index.tolist(),
            'test_assignment': test_assignment.tolist(),
            'status': 'designed'
        }
        
        self.experiments[f"{cluster_id}_{strategy_name}"] = experiment_design
        return experiment_design
    
    def power_analysis(self, effect_size=0.1, alpha=0.05, power=0.8):
        """Calculate required sample size"""
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return {
            'sample_size_per_group': int(np.ceil(n_per_group)),
            'total_sample_size': int(np.ceil(2 * n_per_group)),
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power
        }
    
    def simulate_test_results(self, experiment_key, treatment_effect=0.1, 
                            metric='revenue_per_customer'):
        """Simulate A/B test results"""
        
        if experiment_key not in self.experiments:
            raise ValueError(f"Experiment {experiment_key} not found")
        
        exp = self.experiments[experiment_key]
        cluster_data = self.data[self.data['cluster'] == exp['cluster_id']]
        
        # Baseline metric
        if metric == 'revenue_per_customer':
            baseline = (cluster_data['PURCHASES'] + cluster_data['CASH_ADVANCE']).mean()
        elif metric == 'utilization':
            baseline = cluster_data['effective_utilization'].mean()
        else:
            baseline = cluster_data[metric].mean()
        
        # Simulate results
        control_values = np.random.normal(baseline, baseline * 0.2, exp['control_group_size'])
        test_values = np.random.normal(baseline * (1 + treatment_effect), 
                                     baseline * 0.2, exp['test_group_size'])
        
        results = {
            'control_mean': np.mean(control_values),
            'test_mean': np.mean(test_values),
            'control_std': np.std(control_values),
            'test_std': np.std(test_values),
            'lift': (np.mean(test_values) - np.mean(control_values)) / np.mean(control_values),
            'raw_values': {'control': control_values, 'test': test_values}
        }
        
        return results
    
    def statistical_test(self, results, test_type='ttest'):
        """Perform statistical significance test"""
        
        control_data = results['raw_values']['control']
        test_data = results['raw_values']['test']
        
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_ind(test_data, control_data)
        elif test_type == 'mannwhitney':
            statistic, p_value = stats.mannwhitneyu(test_data, control_data, 
                                                    alternative='two-sided')
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Confidence interval for difference
        pooled_std = np.sqrt(((len(control_data)-1)*results['control_std']**2 + 
                            (len(test_data)-1)*results['test_std']**2) / 
                            (len(control_data) + len(test_data) - 2))
        
        se_diff = pooled_std * np.sqrt(1/len(control_data) + 1/len(test_data))
        diff = results['test_mean'] - results['control_mean']
        
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': diff / results['control_mean']
        }
    
    def experiment_dashboard(self, experiment_key):
        """Generate experiment monitoring dashboard"""
        
        if experiment_key not in self.experiments:
            raise ValueError(f"Experiment {experiment_key} not found")
        
        exp = self.experiments[experiment_key]
        
        # Simulate results for dashboard
        results = self.simulate_test_results(experiment_key)
        stats_test = self.statistical_test(results)
        
        dashboard = {
            'experiment_info': {
                'name': experiment_key,
                'cluster': exp['cluster_id'],
                'status': exp['status'],
                'duration': exp['duration_days'],
                'sample_sizes': {
                    'control': exp['control_group_size'],
                    'test': exp['test_group_size']
                }
            },
            'results': {
                'control_metric': results['control_mean'],
                'test_metric': results['test_mean'],
                'lift': results['lift'],
                'lift_pct': results['lift'] * 100,
                'statistical_significance': stats_test['significant'],
                'p_value': stats_test['p_value'],
                'confidence_interval': stats_test['confidence_interval']
            },
            'recommendation': self._generate_recommendation(results, stats_test)
        }
        
        return dashboard
    
    def _generate_recommendation(self, results, stats_test):
        """Generate test recommendation"""
        
        if not stats_test['significant']:
            return "Continue test - no significant difference detected"
        
        if results['lift'] > 0.05:  # 5% improvement
            return "Launch strategy - significant positive impact"
        elif results['lift'] < -0.02:  # 2% decline
            return "Stop test - negative impact detected"
        else:
            return "Inconclusive - effect too small for business impact"
    
    def multiple_testing_correction(self, experiment_keys, method='bonferroni'):
        """Correct for multiple testing"""
        
        p_values = []
        for key in experiment_keys:
            results = self.simulate_test_results(key)
            stats_test = self.statistical_test(results)
            p_values.append(stats_test['p_value'])
        
        if method == 'bonferroni':
            corrected_alpha = 0.05 / len(p_values)
            significant = [p < corrected_alpha for p in p_values]
        elif method == 'fdr':
            # Benjamini-Hochberg procedure
            sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
            significant = [False] * len(p_values)
            
            for i, (orig_idx, p_val) in enumerate(sorted_p):
                threshold = (i + 1) / len(p_values) * 0.05
                if p_val <= threshold:
                    significant[orig_idx] = True
        
        return {
            'original_p_values': p_values,
            'significant_after_correction': significant,
            'method': method,
            'corrected_alpha': corrected_alpha if method == 'bonferroni' else None
        }

## Usage example
# ab_test = ABTestingFramework(df_risk_final, cluster_labels)
# experiment = ab_test.design_experiment(cluster_id=0, strategy_name='retention_program')
# results = ab_test.experiment_dashboard('0_retention_program')