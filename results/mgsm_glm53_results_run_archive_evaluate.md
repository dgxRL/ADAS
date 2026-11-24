# MGSM GLM-5.3 — Evaluate

## Chain-of-Thought

By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.

- **generation**: initial
- **fitness**: 95% Bootstrap Confidence Interval: (93.8%, 99.2%), Median: 96.9%
- **test_fitness**: 95% Bootstrap Confidence Interval: (0.0%, 0.0%), Median: 0.0%

```python
def forward(self, taskInfo):
    # Instruction for the Chain-of-Thought (CoT) approach
    # It is an important practice that allows the LLM to think step by step before solving the task.
    cot_instruction = "Please think step by step and then solve the task."

    # Instantiate a new LLM agent specifically for CoT
    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # Prepare the inputs for the CoT agent
    # The input should be a list of Info, and the first one is often the taskInfo
    cot_agent_inputs = [taskInfo]

    # Get the response from the CoT agent
    thinking, answer = cot_agent(cot_agent_inputs, cot_instruction)

    # Return only the final answer
    return answer

```

## Self-Consistency with Chain-of-Thought

While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final answer through ensembling.

- **generation**: initial
- **fitness**: 95% Bootstrap Confidence Interval: (96.1%, 100.0%), Median: 98.4%
- **test_fitness**: 95% Bootstrap Confidence Interval: (0.0%, 0.0%), Median: 0.0%

```python
def forward(self, taskInfo):
    # Instruction for step-by-step reasoning
    cot_instruction = "Please think step by step and then solve the task."
    N = 5 # Number of CoT agents

    # Initialize multiple CoT agents with a higher temperature for varied reasoning
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]

    # Majority voting function to select the most common answer
    from collections import Counter
    def majority_voting(answers):
        return Counter(answers).most_common(1)[0][0]
    
    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.append(answer.content)

    # Ensembling the answers from multiple CoT agents
    answer = majority_voting(possible_answers)
    return answer  

```

## Self-Refine (Reflexion)

To enhance its performance, an LLM can iteratively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution.

- **generation**: initial
- **fitness**: 95% Bootstrap Confidence Interval: (94.5%, 100.0%), Median: 97.7%
- **test_fitness**: 95% Bootstrap Confidence Interval: (79.6%, 87.1%), Median: 83.3%

```python
def forward(self, taskInfo):
    # Instruction for initial reasoning
    cot_initial_instruction = "Please think step by step and then solve the task."

    # Instruction for reflecting on previous attempts and feedback to improve
    cot_reflect_instruction = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # Instruction for providing feedback and correcting the answer
    critic_instruction = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output 'True' in 'correct'."
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent')
    
    N_max = 5 # Maximum number of attempts

    # Initial attempt
    cot_inputs = [taskInfo]
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        # Get feedback and correct status from the critic
        feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break
            
        # Add feedback to the inputs for the next iteration
        cot_inputs.extend([thinking, answer, feedback])

        # Reflect on previous attempts and refine the answer
        thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1)
    return answer

```
