# models/baseline.py
"""
Baseline Ensemble Model for Clinical Note Classification

Theory-driven routing ensemble combining:
1. Regex patterns (domain knowledge)
2. Baseline classifier (trained on balanced data)
3. Semi-supervised classifier (trained on expanded data)

Routing logic leverages each component's strengths based on training characteristics.
"""

import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class BaselineEnsemble:
    """
    Theory-driven routing ensemble for clinical classification
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize baseline ensemble
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters (random_state, thresholds, etc.)
        """
        self.config = config or {}
        self.random_state = self.config.get('random_state', 42)
        
        # Regex patterns
        self.diabetes_patterns = (
            r'\bT2DM\b|\bT1DM\b|'
            r'\btype\s*[12]\s*diabetes(?:\s*mellitus)?\b|'
            r'\bdiabetes\s*mellitus\b|'
            r'\bhistory\s*of\s*diabetes\b|'
            r'\bdiabetic\b|'
            r'\binsulin\s*(?:aspart|therapy|dependent|glargine|lispro)\b|'
            r'\bmetformin\b|\bglipizide\b|\bglyburide\b'
        )
        
        self.cancer_patterns = (
            r'\bcancer\b|\bcarcinoma\b|\bsarcoma\b|\bmelanoma\b|'
            r'\blymphoma\b|\bleukemia\b|\bmalignancy\b|\bmalignant\b|'
            r'\bneoplasm\b|\bchemotherapy\b|\boncolog(?:y|ist)\b|'
            r'\bmetastatic\b|\btumor\b|\bMDS\s*with\s*excess\s*blasts\b'
        )
        
        # Models
        self.clf_baseline = None
        self.clf_semisup = None
        self.classes_ = None
        
        # Binary classifiers for synthetic labeling
        self.clf_diabetes_binary = None
        self.clf_cancer_binary = None
        
        # Training metadata
        self.n_baseline_train = None
        self.n_semisup_train = None
        self.baseline_neither_pct = None
        self.n_labeled = None
        self.n_synthetic = None
    
    
    def create_stratified_split(
        self,
        df: pd.DataFrame,
        get_combined_label_fn
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/test split with oversampling of rare classes
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataframe with all data
        get_combined_label_fn : callable
            Function to create combined labels from has_cancer/has_diabetes
        
        Returns:
        --------
        train_labeled : pd.DataFrame
            Training set with labels
        test_labeled : pd.DataFrame
            Test set with labels
        """
        # Get all labeled examples
        all_labeled = df[(df['test_set'] == 0) & 
                         (df['has_cancer'].notna()) & 
                         (df['has_diabetes'].notna())].copy()
        all_labeled['combined_label'] = all_labeled.apply(get_combined_label_fn, axis=1)
        
        # Sampling strategy with oversampling of rare classes
        sampling_strategy = {
            'Neither': 0.25,
            'Cancer Only': 0.30,  
            'Diabetes Only': 0.67,
            'Both': 0.50
        }
        
        test_samples = []
        train_samples = []
        
        for label, test_fraction in sampling_strategy.items():
            label_data = all_labeled[all_labeled['combined_label'] == label]
            n_total = len(label_data)
            n_test = max(1, int(n_total * test_fraction))
            
            label_shuffled = label_data.sample(frac=1, random_state=self.random_state)
            test_samples.append(label_shuffled.iloc[:n_test])
            train_samples.append(label_shuffled.iloc[n_test:])
        
        train_labeled = pd.concat(train_samples, ignore_index=True)
        test_labeled = pd.concat(test_samples, ignore_index=True)
        
        return train_labeled, test_labeled
    
    
    def _train_binary_classifiers(
        self,
        X_train: np.ndarray,
        y_diabetes: np.ndarray,
        y_cancer: np.ndarray
    ):
        """
        Train binary classifiers for synthetic label generation
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_diabetes : np.ndarray
            Binary diabetes labels
        y_cancer : np.ndarray
            Binary cancer labels
        """
        self.clf_diabetes_binary = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state
        )
        self.clf_diabetes_binary.fit(X_train, y_diabetes)
        
        self.clf_cancer_binary = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state
        )
        self.clf_cancer_binary.fit(X_train, y_cancer)
    
    def generate_synthetic_labels(
        self,
        df: pd.DataFrame,
        train_labeled: pd.DataFrame,
        test_labeled: pd.DataFrame,
        get_combined_label_fn
    ) -> pd.DataFrame:
        """
        Generate synthetic labels using three-phase approach:
        Phase 1: ML-based (high confidence predictions)
        Phase 2: Regex-based (keyword matching)
        Phase 3: ML-based 'Neither' (low disease probability)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataframe
        train_labeled : pd.DataFrame
            Labeled training data
        test_labeled : pd.DataFrame
            Labeled test data
        get_combined_label_fn : callable
            Function to create combined labels from has_cancer/has_diabetes
        
        Returns:
        --------
        final_expanded : pd.DataFrame
            Labeled data + synthetic labels
        """
        # Check if embeddings exist
        if 'embeddings' not in train_labeled.columns:
            raise ValueError("Embeddings not found in train_labeled. Run load_data with generate_embeddings=True")
        
        # Train binary classifiers
        X_train = np.vstack(train_labeled['embeddings'].values)
        y_diabetes = train_labeled['has_diabetes'].values
        y_cancer = train_labeled['has_cancer'].values
        
        self._train_binary_classifiers(X_train, y_diabetes, y_cancer)
        print("✓ Binary classifiers trained for synthetic label generation")
        
        # Get unlabeled data (excluding train and test)
        train_labeled_ids = set(train_labeled['patient_identifier'].values)
        test_labeled_ids = set(test_labeled['patient_identifier'].values)
        
        unlabeled_data = df[
            (df['test_set'] == 0) & 
            (df['has_cancer'].isna()) & 
            (df['has_diabetes'].isna()) &
            (~df['patient_identifier'].isin(train_labeled_ids)) &
            (~df['patient_identifier'].isin(test_labeled_ids))
        ].copy()
        
        print(f"\nUnlabeled data: {len(unlabeled_data)} examples")
        
        # PHASE 1: ML-based synthetic labeling
        ml_synthetic = self._phase1_ml_synthetic(unlabeled_data, get_combined_label_fn)
        
        # PHASE 2: Regex-based synthetic labeling
        regex_synthetic = self._phase2_regex_synthetic(
            unlabeled_data, ml_synthetic, get_combined_label_fn
        )
        
        # PHASE 3: ML-based 'Neither' labels
        ml_synthetic_neither = self._phase3_neither_synthetic(
            unlabeled_data, ml_synthetic, regex_synthetic
        )
        
        # Combine all data
        train_labeled_copy = train_labeled.copy()
        train_labeled_copy['source'] = 'labeled'
        
        all_dfs = [train_labeled_copy]
        if len(ml_synthetic) > 0:
            all_dfs.append(ml_synthetic)
        if len(regex_synthetic) > 0:
            all_dfs.append(regex_synthetic)
        if len(ml_synthetic_neither) > 0:
            all_dfs.append(ml_synthetic_neither)
        
        final_expanded = pd.concat(all_dfs, ignore_index=True)
        
        return final_expanded

    
    
    def _phase1_ml_synthetic(
        self,
        unlabeled_data: pd.DataFrame,
        get_combined_label_fn
    ) -> pd.DataFrame:
        """
        Phase 1: Generate synthetic labels using ML with 60% confidence threshold
        """
        ml_synthetic = pd.DataFrame()
        
        if len(unlabeled_data) > 0:
            X_unlabeled = np.vstack(unlabeled_data['embeddings'].values)
            
            prob_diabetes = self.clf_diabetes_binary.predict_proba(X_unlabeled)[:, 1]
            prob_cancer = self.clf_cancer_binary.predict_proba(X_unlabeled)[:, 1]
            
            # 60% confidence threshold for positive, <20% for neither
            high_conf_diabetes = prob_diabetes >= 0.6
            high_conf_cancer = prob_cancer >= 0.6
            high_conf_neither = (prob_diabetes < 0.2) & (prob_cancer < 0.2)
            
            high_conf_mask = high_conf_diabetes | high_conf_cancer | high_conf_neither
            
            if high_conf_mask.sum() > 0:
                ml_synthetic = unlabeled_data[high_conf_mask].copy()
                ml_synthetic['has_diabetes'] = (prob_diabetes[high_conf_mask] >= 0.6).astype(int)
                ml_synthetic['has_cancer'] = (prob_cancer[high_conf_mask] >= 0.6).astype(int)
                ml_synthetic['combined_label'] = ml_synthetic.apply(get_combined_label_fn, axis=1)
                ml_synthetic['source'] = 'ml_synthetic'
                
                print(f"Phase 1 - ML synthetic: {len(ml_synthetic)} examples")
                print(f"  Distribution: {ml_synthetic['combined_label'].value_counts().to_dict()}")
        
        return ml_synthetic
    
    
    def _phase2_regex_synthetic(
        self,
        unlabeled_data: pd.DataFrame,
        ml_synthetic: pd.DataFrame,
        get_combined_label_fn
    ) -> pd.DataFrame:
        """
        Phase 2: Generate synthetic labels using regex keyword matching
        """
        regex_synthetic = pd.DataFrame()
        
        # Get remaining unlabeled after Phase 1
        if len(ml_synthetic) > 0:
            ml_synthetic_indices = set(ml_synthetic.index)
            remaining_after_ml = unlabeled_data[~unlabeled_data.index.isin(ml_synthetic_indices)].copy()
        else:
            remaining_after_ml = unlabeled_data.copy()
        
        if len(remaining_after_ml) > 0:
            remaining_after_ml['has_diabetes_regex'] = remaining_after_ml['text'].str.contains(
                self.diabetes_patterns, case=False, regex=True, na=False
            ).astype(int)
            
            remaining_after_ml['has_cancer_regex'] = remaining_after_ml['text'].str.contains(
                self.cancer_patterns, case=False, regex=True, na=False
            ).astype(int)
            
            regex_synthetic = remaining_after_ml[
                (remaining_after_ml['has_diabetes_regex'] == 1) | 
                (remaining_after_ml['has_cancer_regex'] == 1)
            ].copy()
            
            regex_synthetic['has_diabetes'] = regex_synthetic['has_diabetes_regex']
            regex_synthetic['has_cancer'] = regex_synthetic['has_cancer_regex']
            regex_synthetic['combined_label'] = regex_synthetic.apply(get_combined_label_fn, axis=1)
            regex_synthetic['source'] = 'regex'
            
            print(f"Phase 2 - Regex synthetic: {len(regex_synthetic)} examples")
            print(f"  Distribution: {regex_synthetic['combined_label'].value_counts().to_dict()}")
        
        return regex_synthetic
    
    
    def _phase3_neither_synthetic(
        self,
        unlabeled_data: pd.DataFrame,
        ml_synthetic: pd.DataFrame,
        regex_synthetic: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Phase 3: Generate 'Neither' synthetic labels with relaxed threshold
        """
        ml_synthetic_neither = pd.DataFrame()
        
        # Get remaining unlabeled after Phase 1 & 2
        used_indices = set()
        if len(ml_synthetic) > 0:
            used_indices.update(ml_synthetic.index)
        if len(regex_synthetic) > 0:
            used_indices.update(regex_synthetic.index)
        
        if len(unlabeled_data) > 0:
            remaining_after_phase2 = unlabeled_data[~unlabeled_data.index.isin(used_indices)].copy()
            
            if len(remaining_after_phase2) > 0:
                X_remaining = np.vstack(remaining_after_phase2['embeddings'].values)
                
                prob_diabetes_neither = self.clf_diabetes_binary.predict_proba(X_remaining)[:, 1]
                prob_cancer_neither = self.clf_cancer_binary.predict_proba(X_remaining)[:, 1]
                
                # RELAXED threshold: both diseases <30%
                neither_mask = (prob_diabetes_neither < 0.3) & (prob_cancer_neither < 0.3)
                
                if neither_mask.sum() > 0:
                    # Target ~200 'Neither' examples
                    target_neither_count = min(neither_mask.sum(), 200)
                    
                    ml_synthetic_neither = remaining_after_phase2[neither_mask].sample(
                        n=target_neither_count,
                        random_state=self.random_state
                    ).copy()
                    
                    ml_synthetic_neither['has_diabetes'] = 0
                    ml_synthetic_neither['has_cancer'] = 0
                    ml_synthetic_neither['combined_label'] = 'Neither'
                    ml_synthetic_neither['source'] = 'ml_neither'
                    
                    print(f"Phase 3 - ML 'Neither': {len(ml_synthetic_neither)} examples")
        
        return ml_synthetic_neither
    
    
    def train(
        self,
        X_baseline: np.ndarray,
        y_baseline: np.ndarray,
        X_semisup: np.ndarray,
        y_semisup: np.ndarray
    ):
        """
        Train both baseline and semi-supervised models
        
        Parameters:
        -----------
        X_baseline : np.ndarray
            Features for baseline model (original labeled data)
        y_baseline : np.ndarray
            Labels for baseline model
        X_semisup : np.ndarray
            Features for semi-supervised model (expanded data)
        y_semisup : np.ndarray
            Labels for semi-supervised model
        """
        # Train baseline model
        self.clf_baseline = LogisticRegression(
            class_weight='balanced',
            C=1.0,
            max_iter=1000,
            random_state=self.random_state
        )
        self.clf_baseline.fit(X_baseline, y_baseline)
        
        # Train semi-supervised model
        self.clf_semisup = LogisticRegression(
            class_weight='balanced',
            C=1.0,
            max_iter=1000,
            random_state=self.random_state
        )
        self.clf_semisup.fit(X_semisup, y_semisup)
        
        # Store metadata
        self.classes_ = self.clf_baseline.classes_
        self.n_baseline_train = len(y_baseline)
        self.n_semisup_train = len(y_semisup)
        self.n_labeled = len(y_baseline)
        self.n_synthetic = len(y_semisup) - len(y_baseline)
        self.baseline_neither_pct = (y_baseline == 'Neither').sum() / len(y_baseline)
    
    
    def predict(
        self,
        X_test: np.ndarray,
        texts_test: List[str]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Predict using routing ensemble logic
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        texts_test : list of str
            Original text for regex matching
        
        Returns:
        --------
        predictions : np.ndarray
            Predicted labels
        decision_log : list of dict
            Detailed reasoning for each prediction
        """
        if self.clf_baseline is None or self.clf_semisup is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Get predictions from both models
        y_pred_baseline = self.clf_baseline.predict(X_test)
        y_pred_baseline_proba = self.clf_baseline.predict_proba(X_test)
        
        y_pred_semisup = self.clf_semisup.predict(X_test)
        y_pred_semisup_proba = self.clf_semisup.predict_proba(X_test)
        
        predictions = []
        decision_log = []
        
        for i in range(len(X_test)):
            pred, log = self._predict_single(
                baseline_pred=y_pred_baseline[i],
                baseline_proba=y_pred_baseline_proba[i],
                semisup_pred=y_pred_semisup[i],
                semisup_proba=y_pred_semisup_proba[i],
                text=texts_test[i],
                case_num=i + 1
            )
            predictions.append(pred)
            decision_log.append(log)
        
        return np.array(predictions), decision_log
    
    
    def _predict_single(
        self,
        baseline_pred: str,
        baseline_proba: np.ndarray,
        semisup_pred: str,
        semisup_proba: np.ndarray,
        text: str,
        case_num: int
    ) -> Tuple[str, Dict]:
        """
        Predict single case using routing logic
        
        Routing tiers:
        1. REGEX: Explicit keywords override ML
        2. AGREEMENT: Both models agree
        3. MODEL STRENGTHS: Route based on training characteristics
        4. WEIGHTED VOTING: Fallback
        """
        baseline_probs = dict(zip(self.classes_, baseline_proba))
        semisup_probs = dict(zip(self.classes_, semisup_proba))
        
        # Check regex
        has_diabetes_kw = bool(re.search(self.diabetes_patterns, text, re.IGNORECASE))
        has_cancer_kw = bool(re.search(self.cancer_patterns, text, re.IGNORECASE))
        
        # TIER 1: REGEX
        if has_diabetes_kw and has_cancer_kw:
            prediction = 'Both'
            reason = "REGEX: Both keywords found"
            confidence = 1.0
            
        elif has_diabetes_kw:
            prediction = 'Diabetes Only'
            reason = "REGEX: Diabetes keyword found"
            confidence = 1.0
            
        elif has_cancer_kw:
            prediction = 'Cancer Only'
            reason = "REGEX: Cancer keyword found"
            confidence = 1.0
        
        # TIER 2: AGREEMENT
        elif baseline_pred == semisup_pred:
            prediction = baseline_pred
            confidence = (baseline_probs[baseline_pred] + semisup_probs[semisup_pred]) / 2
            reason = f"AGREEMENT: Both predict {prediction}"
        
        # TIER 3: MODEL STRENGTHS
        elif baseline_pred == 'Neither':
            prediction = 'Neither'
            confidence = baseline_probs['Neither']
            reason = "BASELINE strength: Trained on balanced 'Neither' data"
            
        elif semisup_pred in ['Cancer Only', 'Diabetes Only', 'Both']:
            prediction = semisup_pred
            confidence = semisup_probs[semisup_pred]
            reason = "SEMI-SUP strength: More training data for diseases"
        
        # TIER 4: WEIGHTED VOTING
        else:
            votes = {}
            baseline_weight = 1.5 if baseline_pred == 'Neither' else 1.0
            votes[baseline_pred] = baseline_probs[baseline_pred] * baseline_weight
            
            semisup_weight = 2.0 if semisup_pred != 'Neither' else 1.0
            votes[semisup_pred] = semisup_probs[semisup_pred] * semisup_weight
            
            prediction = max(votes, key=votes.get)
            confidence = votes[prediction] / (baseline_weight + semisup_weight)
            reason = f"WEIGHTED VOTE: {prediction} wins"
        
        # Build decision log
        log = {
            'case_num': case_num,
            'prediction': prediction,
            'reason': reason,
            'confidence': confidence,
            'baseline_pred': baseline_pred,
            'baseline_conf': baseline_probs[baseline_pred],
            'semisup_pred': semisup_pred,
            'semisup_conf': semisup_probs[semisup_pred],
            'has_diabetes_kw': has_diabetes_kw,
            'has_cancer_kw': has_cancer_kw
        }
        
        return prediction, log
    
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        texts_test: List[str]
    ) -> Dict:
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            True labels
        texts_test : list of str
            Original text
        
        Returns:
        --------
        results : dict
            Evaluation metrics and predictions
        """
        predictions, decision_log = self.predict(X_test, texts_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions, labels=self.classes_)
        
        # Per-class recall
        per_class_recall = {}
        for cls in self.classes_:
            mask = y_test == cls
            if mask.sum() > 0:
                correct = ((y_test == cls) & (predictions == cls)).sum()
                per_class_recall[cls] = correct / mask.sum()
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'per_class_recall': per_class_recall,
            'predictions': predictions,
            'decision_log': decision_log
        }
        
        return results
    
    
    def print_ensemble_strategy(self):
        """Print detailed ensemble strategy and justification"""
        print("\n" + "="*80)
        print("ENSEMBLE STRATEGY (PRE-SPECIFIED)")
        print("="*80)
        print("JUSTIFICATION: Each model has theoretical strengths based on training:")
        print("")
        print("1. REGEX (Highest Priority)")
        print("   Rationale: High precision when explicit medical keywords present")
        print("   Use: Override ML when diabetes/cancer keywords detected")
        print("   Evidence: From Weak Supervision analysis")
        print("")
        print("2. BASELINE for 'Neither'")
        neither_count = int(self.n_baseline_train * self.baseline_neither_pct) if self.n_baseline_train else 0
        print(f"   Rationale: Trained on balanced data ({neither_count}/{self.n_baseline_train} = {self.baseline_neither_pct:.0%} 'Neither')")
        print("   Use: When baseline predicts 'Neither', it has seen balanced examples")
        print("")
        print("3. SEMI-SUPERVISED for Positive Cases")
        expansion_factor = self.n_semisup_train / self.n_baseline_train if self.n_baseline_train else 0
        print(f"   Rationale: {expansion_factor:.0f}x more training data for disease patterns")
        print("   Use: When predicting Cancer/Diabetes/Both")
        print("")
        print("4. AGREEMENT")
        print("   Rationale: High confidence when both models agree")
        print("   Use: Average probabilities when predictions match")
        print("")
        print("5. WEIGHTED VOTING (Last Resort)")
        print("   Rationale: Balance model strengths when they disagree")
        print("   Weights: Baseline=1.5x for 'Neither', Semi-sup=2.0x for positive")
        print("="*80)
    
    
    def print_methodology(self):
        """Print ensemble methodology summary"""
        print("\n" + "="*80)
        print("THEORY-DRIVEN ROUTING ENSEMBLE")
        print("="*80)
        
        if self.n_baseline_train:
            print(f"\nTraining Data:")
            print(f"  Baseline: {self.n_baseline_train} examples (original labeled)")
            print(f"  Semi-supervised: {self.n_semisup_train} examples ({self.n_labeled} labeled + {self.n_synthetic} synthetic)")
            print(f"  Baseline 'Neither': {self.baseline_neither_pct:.0%}")
        
        print("\nRouting Strategy:")
        print("  1. REGEX - Override when keywords present")
        print("  2. AGREEMENT - Use when models agree")
        print("  3. MODEL STRENGTHS - Route to specialized model")
        print("  4. WEIGHTED VOTING - Fallback with balanced weights")
        print("="*80)