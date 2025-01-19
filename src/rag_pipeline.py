
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re
import pandas as pd
from tqdm import tqdm
import dotenv
import os

dotenv.load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

df = pd.read_csv("data/climate_promise_constraint.csv")
question = "Does the sentence contain any claim that mentions constraints, impediments or goals, targets connected with sustainability, net-zero, environmental or sustainability targets?"


prompt = PromptTemplate(
    template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a helpful assistant who assists human analysts in answering a question regarding the text. 
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    
    Question: {question}
    Sentence: {sentence}
    Assign:
    - 'Promise' - a forward-looking claim that tends to lack specific, measurable actions or mechanisms for accountability. These claims frequently rely on optimistic language, future simple and present continuous tenses, stating that some actions are either in progress or planned without defining specific criteria for their execution or time periods.
    - 'Constraint' - a sentence that mentions some impediments, restrictions, or obstacles at a company, society, or governmental level that may restrict the company from fulfilling its promises. You do not need to consider descriptions of risks connected with climate change as constraints unless it is explicitly stated that some climate issues now influence a company's performance or have a visible impact on its operational abilities.
    - 'None' if none of the above is applicable.
    
    ## Example 1:
    Question: Does the sentence contain any claim that mentions constraints, impediments or goals, targets connected with sustainability, net-zero, environmental or sustainability targets?
    Sentence: Disruption of our supply chain, including increased commodity, raw material, packaging, energy, transportation, and other input costs.
    [Guess]: Contradiction
    [Confidence]: 0.8
    
    ## Example 2:
    Question: Does the sentence contain any claim that mentions constraints, impediments or goals, targets connected with sustainability, net-zero, environmental or sustainability targets?
    Sentence: We are committed to achieving net-zero carbon emissions by 2050.
    [Guess]: Promise
    [Confidence]: 0.9
    
    ## Example 3:
    Question: Does the sentence contain any claim that mentions constraints, impediments or goals, targets connected with sustainability, net-zero, environmental or sustainability targets?
    Sentence: The negative impacts of, and continuing uncertainties associated with the scope, severity, and duration of the global COVID-19 pandemic and the substance and pace of the post-pandemic economic recovery.
    [Guess]: Contradiction
    [Confidence]: 0.7

    Reply in the following format:
    [Guess]: <Your most likely guess, should be [Promise, Contradiction, None].>
    [Confidence]: <Give your honest confidence score between 0.0 and 1.0 about the correctness of your guess. 0 means your previous guess is very likely to be wrong, and 1 means you are very confident about the guess.>
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "sentence"]
)

# Parse the input string to extract the guess and confidence values
def parse_input(input_string):
    guess_match = re.search(r'\[Guess\]:\s*(.+)', input_string)
    confidence_match = re.search(r'\[Confidence\]:\s*(.+)', input_string)
    # Get the matched values or return None if not found
    guess = guess_match.group(1).strip() if guess_match else None
    confidence = float(confidence_match.group(1).strip()) if confidence_match else None
    return guess, confidence


class RAG:
    def __init__(self, prompt):
        self.prompt = prompt

    def score_text_claim_batch(self, questions, texts):
        """
        Scores each text in `texts` for its relevance to the question & explanation.
        Returns a list of outputs from the model. If an error is encountered, raise it
        so the calling code can handle it and track which batch caused the error.
        """
        rag_chain = self.prompt | LLM | StrOutputParser()

        # Create a batch of inputs
        batch_inputs = [
            {"question": questions[i], "sentence": texts[i]} 
            for i in range(len(texts))
        ]
        # Invoke the chain for all inputs in a batch
        generations = [rag_chain.invoke(input_data) for input_data in batch_inputs]
        return generations


df_extended = df.copy()
if 'score' not in df_extended.columns:
    df_extended.loc[:, 'score_prompt'] = None  
if 'prediction' not in df_extended.columns:
    df_extended.loc[:, 'prediction_prompt'] = None  


rag = RAG(prompt)
batch_size = 32
# Create a separate DataFrame to store problematic batches
error_df = pd.DataFrame(columns=df_extended.columns.tolist() + ["error_message"])

# Initialize a counter to track the number of processed batches
batch_counter = 0

for batch_start in tqdm(range(0, df_extended.shape[0], batch_size), 
                        total=(df_extended.shape[0] // batch_size) + 1):
    batch_end = min(batch_start + batch_size, df_extended.shape[0])
    
    # Prepare batch inputs
    batch_questions = [question] * (batch_end - batch_start)
    batch_texts = df_extended.iloc[batch_start:batch_end]["sentence"].tolist()

    try:
        # Process the batch
        batch_results = rag.score_text_claim_batch(batch_questions, batch_texts)
    except Exception as e:
        # Handle errors by adding the problematic batch to error_df
        error_batch = df_extended.iloc[batch_start:batch_end].copy()
        error_batch["error_message"] = str(e)
        error_df = pd.concat([error_df, error_batch], ignore_index=True)
        # Continue to the next batch
        continue

    # Update the DataFrame with results
    for i, result in enumerate(batch_results):
        guess, confidence = parse_input(result)
        df_extended.loc[batch_start + i, "score_prompt2"] = confidence
        df_extended.loc[batch_start + i, "prediction_prompt2"] = guess

    # Increment the batch counter
    batch_counter += 1

    # Log partial results after every 10 batches
    if batch_counter % 10 == 0:
        partial_csv_path = f"manufacturing_distilled_positive_{batch_counter}.csv"
        df_extended.to_csv(partial_csv_path, index=False)
        print(f"Logged partial results after {batch_counter} batches to {partial_csv_path}")

# Save the final results and errors if applicable
df_extended.to_csv("final_data.csv", index=False)
if not error_df.empty:
    error_df.to_csv("error_batches.csv", index=False)
    print("Saved error batches to error_batches.csv")


