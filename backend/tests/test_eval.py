import sys
import os

from ragas import evaluate, EvaluationDataset
from ragas.models import LangchainLLMWrapper, LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.dataset_schema import SingleTurnSample

from langchain_xai import ChatXAI
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings

# Append backend directory to path to resolve app imports during external CI triggers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.config import settings 

def execute_ci_validation():
    # 1. Initialize your evaluation models
    eval_llm = ChatXAI(model="grok-2", xai_api_key=settings.XAI_API_KEY, temperature=0)
    eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Wrap them in the official Ragas interface schemas
    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)
    
    # 2. Instantiate and configure your evaluation judges
    # Note: Ragas renamed answer_relevance to ResponseRelevancy in newer releases
    faithfulness_metric = Faithfulness(llm=ragas_llm)
    relevance_metric = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb)
    
    # 3. Create the evaluation dataset using the SingleTurnSample paradigm
    samples = [
        SingleTurnSample(
            user_input="What is the coverage rule for deployment pipelines?",
            retrieved_contexts=["Production environments require CI verification pipelines to cross a mandatory 85% coverage bar before merging code updates."],
            response="The deployment framework enforces an 85% coverage threshold score to allow codebase merges.",
            reference="CI validation frameworks require a minimum baseline score of 85% code coverage to release updates."
        )
    ]
    
    dataset = EvaluationDataset(samples=samples)
    
    print("🚀 Initializing Ragas validation across golden sets using Grok-2 judges...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness_metric, relevance_metric]
        )
        
        # Format and convert the result matrix cleanly to a python dict
        scores = result.to_pandas().to_dict(orient="records")[0]
        
        print("\n--- Pipeline Evaluation Dashboard Metrics ---")
        print(f"Faithfulness (Groundedness): {scores.get('faithfulness', 0):.4f}")
        print(f"Answer Relevance:           {scores.get('user_input_similarity', 0):.4f}")
        
        # 4. Gate checking logic
        PASSING_THRESHOLD = 0.75
        
        # Fetching dynamic keys safely out of Ragas pandas conversion tables
        actual_faithfulness = scores.get('faithfulness', 0)
        actual_relevance = scores.get('user_input_similarity', scores.get('answer_relevance', 0))
        
        if actual_faithfulness < PASSING_THRESHOLD or actual_relevance < PASSING_THRESHOLD:
            print("\n❌ Pipeline Failure: System fell below the expected quality bar boundaries.")
            sys.exit(1)
        else:
            print("\n✅ Verification Successful: App metrics meet required production criteria.")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Evaluation broken due to execution error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    execute_ci_validation()