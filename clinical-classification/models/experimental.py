# models/llm_agents.py
"""
LLM-based Three-Agent System for Clinical Note Classification

Combines:
1. Similarity Search Agent (retrieval from vector database)
2. LLM Classification Agent (direct text analysis)
3. Final Decision Agent (synthesis of both approaches)
"""

from openai import OpenAI
import os
import sys
from typing import Dict, List, Tuple
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import load_data, create_train_val_test_splits
from utils.vectore_db_load import create_vector_store, similarity_search, load_vector_store

load_dotenv()


class LLMAgentSystem:
    """
    Three-agent LLM system for clinical classification using OpenAI API
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize LLM agent system
        
        Parameters:
        -----------
        model : str
            OpenAI model to use (default: gpt-4o)
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model
        
        # Load prompts
        self.healthcare_prompt = self._load_healthcare_prompt()
        self.final_decision_prompt = self._load_final_decision_prompt()
    
    
    def _load_healthcare_prompt(self) -> str:
        """Load healthcare classification prompt"""
        return """You are a medical record classification expert specializing in identifying cancer and diabetes diagnoses from hospital discharge summaries.

CRITICAL BUSINESS CONTEXT:
- CANCER: Classify as YES if there is ANY mention of cancer, malignancy, or cancer-like conditions (past, present, under treatment, or in remission)
- DIABETES: Classify as YES ONLY if the patient CURRENTLY has diabetes (not prediabetes or resolved diabetes)

Your task is to classify whether a patient has:
- Cancer (ANY history or current cancer/malignancy)
- Diabetes (CURRENT diabetes only) 
- Both conditions
- Neither condition

=================================================================================
DIABETES DIAGNOSTIC CRITERIA (Source: American Diabetes Association 2024-2025)
=================================================================================

DEFINITIVE DIABETES INDICATORS:
1. Explicit diagnosis statements:
   - "diabetes mellitus", "type 1 diabetes", "type 2 diabetes", "T1DM", "T2DM"
   - "gestational diabetes mellitus" (GDM)
   
2. Laboratory criteria (any ONE of the following):
   - Fasting plasma glucose (FPG) ≥126 mg/dL (7.0 mmol/L)
   - 2-hour plasma glucose ≥200 mg/dL (11.1 mmol/L) during 75-g oral glucose tolerance test
   - HbA1c ≥6.5%
   - Random plasma glucose ≥200 mg/dL (11.1 mmol/L) WITH classic hyperglycemic symptoms

3. Classic hyperglycemic symptoms (when combined with elevated glucose):
   - Polyuria (increased urination) + Polydipsia (increased thirst)
   - Unexplained weight loss
   - These symptoms ALONE are insufficient without confirmed diagnosis or lab values

4. Diabetes-specific treatments:
   - Insulin therapy (aspart, glargine, lispro, NPH, etc.)
   - Oral hypoglycemics: metformin, sulfonylureas, SGLT2 inhibitors, DPP-4 inhibitors
   
5. Diabetes complications (indicate existing diabetes):
   - Diabetic ketoacidosis (DKA)
   - Diabetic retinopathy, nephropathy, or neuropathy
   - Hyperosmolar hyperglycemic state

DO NOT classify as diabetes:
- "Prediabetes" or "impaired fasting glucose" (IFG 100-125 mg/dL)
- "Impaired glucose tolerance" (IGT)
- Transient hyperglycemia (stress, steroids, acute illness)
- "Polyuria and polydipsia" without confirmed diagnosis
- "Rule out diabetes" or "screening for diabetes"

=================================================================================
CANCER DIAGNOSTIC CRITERIA (INCLUSIVE - Business Context)
=================================================================================

CLASSIFY AS CANCER (ANY of the following):
1. Explicit malignancy diagnoses (current or historical):
   - Any cancer type: carcinoma, adenocarcinoma, lymphoma, leukemia, sarcoma, melanoma
   - Specific examples: glioblastoma, myeloma, mesothelioma, neuroblastoma
   - Staging information: TNM staging, Stage 0-IV, AJCC staging
   - "History of [cancer type]" - even if past or in remission

2. Histopathologic confirmation:
   - "Biopsy confirmed malignancy"
   - "Pathology report shows [cancer type]"
   - "Histologic diagnosis of [cancer]"

3. Cancer-specific treatments (past or present):
   - Chemotherapy for malignancy
   - Radiation therapy for cancer
   - Surgical resection of tumor with oncology follow-up
   - Immunotherapy, targeted therapy for cancer

4. Borderline/Cancer-like conditions:
   - Langerhans Cell Histiocytosis (LCH)
   - Histiocytosis
   - Desmoid fibromatosis requiring oncology follow-up
   - Any tumor requiring "oncologist" follow-up or monitoring for "recurrence/metastasis"
   
5. Metastatic disease or cancer remission:
   - "Metastases to [organ]"
   - "Stage IV disease"
   - "Cancer in remission"
   - "Post-cancer treatment follow-up"

6. Oncology involvement:
   - Follow-up with oncologist
   - Oncology monitoring
   - Cancer surveillance imaging

DO NOT classify as cancer:
- Purely benign tumors with NO oncology involvement (e.g., simple cysts, lipomas)
- "Mass" or "lesion" confirmed benign with NO oncology follow-up
- "Rule out cancer" with negative workup
- Prediabetes or metabolic syndrome

=================================================================================
TEMPORAL CONTEXT RULES (DIFFERENT FOR CANCER VS DIABETES)
=================================================================================

CANCER (ANY history = YES):
- "Active malignancy" → YES
- "History of cancer" → YES (even if in remission or resolved)
- "Previous cancer, now in remission" → YES
- "Post-cancer treatment" → YES
- "Cancer surveillance" → YES
- "Undergoing chemotherapy" → YES
- "Past diagnosis of [cancer]" → YES

DIABETES (CURRENT only = YES):
- "Patient has diabetes" → YES
- "History of diabetes" → YES (chronic condition, still present)
- "Known diabetic" → YES
- "Patient's diabetes" → YES
- "On metformin" or other diabetes medications → YES

DIABETES (PAST/RESOLVED = NO):
- "Previous gestational diabetes, resolved after pregnancy" → NO
- "Prediabetes" → NO
- "Impaired glucose tolerance" → NO

UNDER INVESTIGATION (= NO for both):
- "Rule out cancer" → NO
- "Suspected malignancy" (not confirmed) → NO
- "Evaluate for diabetes" → NO

=================================================================================
CLASSIFICATION PROCESS
=================================================================================

1. Read the ENTIRE discharge summary
2. Look for explicit diagnosis statements (primary diagnoses, medical history, problem list)
3. Identify definitive treatments indicating the condition
4. Check for lab values meeting diagnostic criteria
5. Distinguish between current vs. past diagnoses
6. Distinguish between confirmed vs. suspected diagnoses

OUTPUT FORMAT:
Classification: [Cancer Only / Diabetes Only / Both / Neither]
Has Cancer: [0.0 or 1.0]
Has Diabetes: [0.0 or 1.0]
Confidence: [0.0-1.0]
Key Evidence: [Quote 1-2 specific phrases from text that support classification]
Reasoning: [2-3 sentences explaining decision with source-backed criteria]

Now classify this patient discharge summary:

{patient_text}
"""
    
    
    def _load_final_decision_prompt(self) -> str:
        """Load final decision prompt"""
        return """You are a senior medical record analyst making FINAL classification decisions for cancer and diabetes diagnoses.

CRITICAL BUSINESS CONTEXT:
- CANCER: Classify as YES if there is ANY mention of cancer, malignancy, or cancer-like conditions (past, present, under treatment, or in remission)
- DIABETES: Classify as YES ONLY if the patient CURRENTLY has diabetes (not prediabetes or resolved diabetes)

You have been provided with:
1. Similar patient cases from a medical database (RETRIEVAL CONTEXT)
2. An initial AI classification with reasoning and confidence (INITIAL CLASSIFICATION)
3. The current patient's discharge summary (PATIENT TEXT)

Your task is to make the FINAL, DEFINITIVE classification by weighing all available evidence.

=================================================================================
DECISION-MAKING FRAMEWORK
=================================================================================

STEP 1: EVALUATE RETRIEVAL CONTEXT
- How many similar cases have cancer? Diabetes? Both? Neither?
- What is the similarity score of the top matches?
- Do the similar cases provide strong evidence for a pattern?

STEP 2: EVALUATE INITIAL CLASSIFICATION
- What was the AI's classification and confidence?
- Is the reasoning sound based on diagnostic criteria?
- Did it cite specific evidence from the text?

STEP 3: EXAMINE PATIENT TEXT DIRECTLY
- What explicit diagnoses are stated?
- What treatments indicate cancer or diabetes?
- Are there lab values meeting diagnostic thresholds?
- Is the condition current/active vs. past/resolved?

STEP 4: MAKE FINAL DECISION
- If retrieval context and initial classification AGREE → High confidence
- If retrieval context and initial classification DISAGREE → Examine patient text carefully to break tie
- If initial classification has LOW confidence (<0.6) → Give more weight to retrieval context
- If similar cases have MIXED labels → Give more weight to direct text analysis

=================================================================================
DIAGNOSTIC CRITERIA REFERENCE (Business Context)
=================================================================================

DIABETES (CURRENT only):
- Explicit: "diabetes mellitus", "T1DM", "T2DM", "history of diabetes"
- Labs: FPG ≥126 mg/dL, HbA1c ≥6.5%, random glucose ≥200 mg/dL with symptoms
- Treatments: insulin, metformin, sulfonylureas
- Complications: DKA, diabetic retinopathy/nephropathy/neuropathy
- NOT diabetes: "polyuria/polydipsia" alone, prediabetes, transient hyperglycemia, resolved gestational diabetes

CANCER (ANY history):
- Explicit: carcinoma, lymphoma, leukemia, sarcoma, melanoma, with staging
- Confirmation: biopsy-proven malignancy, histopathology
- Treatments: chemotherapy, radiation therapy, surgical resection with oncology follow-up
- Metastases: Stage IV, distant spread, metastases to organs
- Borderline conditions: LCH (Langerhans Cell Histiocytosis), histiocytosis, desmoid fibromatosis
- Oncology involvement: follow-up with oncologist, cancer surveillance
- History: "history of cancer", "cancer in remission", "post-cancer treatment"
- NOT cancer: purely benign tumors with NO oncology involvement, "rule out cancer" with negative results

TEMPORAL CONTEXT:
- CANCER: "History of cancer" or "previous cancer" = YES (any history counts)
- DIABETES: "History of diabetes" = YES (chronic condition, still present)
- DIABETES: "Previous gestational diabetes, resolved" = NO
- "Rule out [condition]" without confirmation = NO

=================================================================================
OUTPUT FORMAT (STRICT - MUST FOLLOW EXACTLY)
=================================================================================

Final Classification: [Cancer Only / Diabetes Only / Both / Neither]
Has Cancer: [0.0 or 1.0]
Has Diabetes: [0.0 or 1.0]
Final Confidence: [0.0-1.0]
Decision Rationale: [3-4 sentences explaining how you weighed the evidence from retrieval, initial classification, and direct text analysis]

=================================================================================
INPUTS FOR CURRENT CASE
=================================================================================

RETRIEVAL CONTEXT (Similar Patient Cases):
{retrieval_context}

INITIAL CLASSIFICATION:
Classification: {initial_classification}
Has Cancer: {initial_has_cancer}
Has Diabetes: {initial_has_diabetes}
Confidence: {initial_confidence}
Reasoning: {initial_reasoning}

PATIENT DISCHARGE SUMMARY:
{patient_text}

Now make your FINAL classification decision:
"""
    
    
    def agent1_similarity_search(
        self,
        patient_text: str,
        vector_store,
        top_k: int = 5,
        collection_name: str = 'patient_embeddings'
    ) -> List[Dict]:
        """
        Agent 1: Retrieve similar patient cases from vector database
        
        Parameters:
        -----------
        patient_text : str
            Patient discharge summary
        vector_store : AstraDBVectorStore
            Initialized vector store (used to get collection name, but fresh store is created)
        top_k : int
            Number of similar cases to retrieve
        collection_name : str
            Collection name for vector store
        
        Returns:
        --------
        similar_cases : list of dict
            Similar patient cases with labels and scores
        """
        
        similar_cases = similarity_search(vector_store, patient_text, top_k=top_k, collection_name=collection_name)
        return similar_cases
    
    
    def agent2_llm_classification(
        self,
        patient_text: str
    ) -> Dict:
        """
        Agent 2: Direct LLM classification of patient text
        
        Parameters:
        -----------
        patient_text : str
            Patient discharge summary
        
        Returns:
        --------
        classification : dict
            Classification with reasoning and confidence
        """
        prompt = self.healthcare_prompt.format(patient_text=patient_text)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical record classification expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        response_text = response.choices[0].message.content
        
        # Parse response
        classification = self._parse_classification_response(response_text)
        return classification
    
    
    def agent3_final_decision(
        self,
        patient_text: str,
        similar_cases: List[Dict],
        initial_classification: Dict
    ) -> Dict:
        """
        Agent 3: Final decision combining retrieval and initial classification
        
        Parameters:
        -----------
        patient_text : str
            Patient discharge summary
        similar_cases : list of dict
            Similar cases from Agent 1
        initial_classification : dict
            Classification from Agent 2
        
        Returns:
        --------
        final_decision : dict
            Final classification with rationale
        """
        # Format retrieval context
        retrieval_context = self._format_retrieval_context(similar_cases)
        
        # Format prompt
        prompt = self.final_decision_prompt.format(
            retrieval_context=retrieval_context,
            initial_classification=initial_classification['classification'],
            initial_has_cancer=initial_classification['has_cancer'],
            initial_has_diabetes=initial_classification['has_diabetes'],
            initial_confidence=initial_classification['confidence'],
            initial_reasoning=initial_classification.get('reasoning', 'N/A'),
            patient_text=patient_text[:2000]  # Limit text length for context window
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior medical record analyst making final classification decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        response_text = response.choices[0].message.content
        
        # Parse response
        final_decision = self._parse_final_decision_response(response_text)
        return final_decision
    
    
    def _format_retrieval_context(self, similar_cases: List[Dict]) -> str:
        """Format similar cases for prompt"""
        context_lines = []
        for i, case in enumerate(similar_cases, 1):
            context_lines.append(f"Similar Case {i} (Similarity: {case['similarity_score']:.3f}):")
            context_lines.append(f"  Label: {case['combined_label']}")
            context_lines.append(f"  Has Cancer: {case['has_cancer']}")
            context_lines.append(f"  Has Diabetes: {case['has_diabetes']}")
            context_lines.append(f"  Text snippet: {case['text'][:200]}...")
            context_lines.append("")
        
        return "\n".join(context_lines)
    
    
    def _parse_classification_response(self, response_text: str) -> Dict:
        """Parse Agent 2 response"""
        lines = response_text.strip().split('\n')
        result = {
            'classification': 'Neither',
            'has_cancer': 0.0,
            'has_diabetes': 0.0,
            'confidence': 0.5,
            'reasoning': ''
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Classification:'):
                result['classification'] = line.split(':', 1)[1].strip()
            elif line.startswith('Has Cancer:'):
                result['has_cancer'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Has Diabetes:'):
                result['has_diabetes'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Confidence:'):
                try:
                    result['confidence'] = float(line.split(':', 1)[1].strip())
                except:
                    result['confidence'] = 0.5
            elif line.startswith('Reasoning:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
        
        return result
    
    
    def _parse_final_decision_response(self, response_text: str) -> Dict:
        """Parse Agent 3 response"""
        lines = response_text.strip().split('\n')
        result = {
            'final_classification': 'Neither',
            'has_cancer': 0.0,
            'has_diabetes': 0.0,
            'final_confidence': 0.5,
            'decision_rationale': ''
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Final Classification:'):
                result['final_classification'] = line.split(':', 1)[1].strip()
            elif line.startswith('Has Cancer:'):
                result['has_cancer'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Has Diabetes:'):
                result['has_diabetes'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Final Confidence:'):
                try:
                    result['final_confidence'] = float(line.split(':', 1)[1].strip())
                except:
                    result['final_confidence'] = 0.5
            elif line.startswith('Decision Rationale:'):
                result['decision_rationale'] = line.split(':', 1)[1].strip()
        
        return result
    
    
    def predict_single(
        self,
        patient_text: str,
        vector_store,
        top_k: int = 5,
        collection_name: str = 'patient_embeddings'
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Run full three-agent pipeline on single patient
        
        Parameters:
        -----------
        patient_text : str
            Patient discharge summary
        vector_store : AstraDBVectorStore
            Vector store for similarity search (used to get collection name)
        top_k : int
            Number of similar cases to retrieve
        collection_name : str
            Collection name for vector store
        
        Returns:
        --------
        similar_cases : list of dict
            Results from Agent 1
        initial_classification : dict
            Results from Agent 2
        final_decision : dict
            Results from Agent 3
        """
        # Agent 1: Similarity search
        similar_cases = self.agent1_similarity_search(patient_text, vector_store, top_k, collection_name=collection_name)
        
        # Agent 2: Initial classification
        initial_classification = self.agent2_llm_classification(patient_text)
        
        # Agent 3: Final decision
        final_decision = self.agent3_final_decision(
            patient_text,
            similar_cases,
            initial_classification
        )
        
        return similar_cases, initial_classification, final_decision