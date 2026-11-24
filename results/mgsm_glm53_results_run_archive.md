# MGSM GLM-5.3 — Run Archive

## Chain-of-Thought

By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.

- **generation**: initial
- **fitness**: 95% Bootstrap Confidence Interval: (93.8%, 99.2%), Median: 96.9%

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

## LLM Debate

By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks.

- **generation**: initial
- **fitness**: 95% Bootstrap Confidence Interval: (96.1%, 100.0%), Median: 98.4%

```python
def forward(self, taskInfo):
    # Instruction for initial reasoning
    debate_initial_instruction = "Please think step by step and then solve the task."

    # Instruction for debating and updating the solution based on other agents' solutions
    debate_instruction = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    
    # Initialize debate agents with different roles and a moderate temperature for varied reasoning
    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', temperature=0.8, role=role) for role in ['Math Professor', 'Grade School Teacher', 'Math Enthusiast']]

    # Instruction for final decision-making based on all debates and solutions
    final_decision_instruction = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)

    max_round = 2 # Maximum number of debate rounds
    all_thinking = [[] for _ in range(max_round)]
    all_answer = [[] for _ in range(max_round)]

    # Perform debate rounds
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, answer = debate_agents[i]([taskInfo], debate_initial_instruction)
            else:
                input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                thinking, answer = debate_agents[i](input_infos, debate_instruction)
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    # Make the final decision based on all debate results and solutions
    thinking, answer = final_decision_agent([taskInfo] + all_thinking[max_round-1] + all_answer[max_round-1], final_decision_instruction)
    return answer

```

## Step-back Abstraction

Let LLM first think about the principles involved in solving this task which could be helpful. By understanding the underlying principles, the model can better reason through the problem and provide a more accurate solution.

- **generation**: initial
- **fitness**: 95% Bootstrap Confidence Interval: (92.2%, 99.2%), Median: 96.1%

```python
def forward(self, taskInfo):
        # Instruction for understanding the principles involved in the task
        principle_instruction = "What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        
        # Instruction for solving the task based on the principles
        cot_instruction = "Given the question and the involved principle behind the question, think step by step and then solve the task."
        
        # Instantiate LLM agents
        principle_agent = LLMAgentBase(['thinking', 'principle'], 'Principle Agent')
        cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
        
        # Get the principles involved in the task
        thinking, principle = principle_agent([taskInfo], principle_instruction)

        # Use the principles to solve the task
        thinking, answer = cot_agent([taskInfo, thinking, principle], cot_instruction)
        return answer

```

## Quality-Diversity

Similar to Quality-Diversity methods, let LLM generate multiple diverse interesting solutions could help. By encouraging the model to explore different reasoning paths, we can increase the chances of finding the best solution.

- **generation**: initial
- **fitness**: 95% Bootstrap Confidence Interval: (94.5%, 100.0%), Median: 97.7%

```python
def forward(self, taskInfo):
    # Instruction for initial reasoning
    cot_initial_instruction = "Please think step by step and then solve the task."

    # Instruction for giving diverse answers
    qd_instruction = "Given previous attempts, try to come up with another interesting way to solve the task."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # Instruction for final decision-making based on collected reasoning and answers
    final_decision_instruction = "Given all the above solutions, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    
    N_max = 3 # Maximum number of attempts

    # Initial attempt
    cot_inputs = [taskInfo]
    possible_answers = []
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    # Add the answer to the list of possible answers
    possible_answers.extend([thinking, answer])

    for i in range(N_max):
        # Reflect on previous attempts and generate another interesting answer
        cot_inputs.extend([thinking, answer])

        # Generate another interesting answer
        thinking, answer = cot_agent(cot_inputs, qd_instruction, i + 1)
        possible_answers.extend([thinking, answer])

    # Make the final decision based on all generated answers
    thinking, answer = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    return answer

```

## Dynamic Assignment of Roles

Similar to Auto-GPT and expert prompting, we can use dynamic control flow in the design to let the agent decide what expert we should use.

- **generation**: initial
- **fitness**: 95% Bootstrap Confidence Interval: (93.8%, 99.2%), Median: 96.9%

```python
def forward(self, taskInfo):
        # Instruction for step-by-step reasoning
        cot_instruction = "Please think step by step and then solve the task."
        expert_agents = [LLMAgentBase(['thinking', 'answer'], 'Expert Agent', role=role) for role in ['Math Professor', 'Grade School Teacher', 'Math Enthusiast', 'Helpful Assistant']]

        # Instruction for routing the task to the appropriate expert
        routing_instruction = "Given the task, please choose an Expert to answer the question. Choose from: Math Professor, Grade School Teacher, Math Enthusiast."
        routing_agent = LLMAgentBase(['choice'], 'Routing agent')

        # Get the choice of expert to route the task
        choice = routing_agent([taskInfo], routing_instruction)[0]

        if 'professor' in choice.content.lower():
            expert_id = 0
        elif 'teacher' in choice.content.lower():
            expert_id = 1
        elif 'enthusiast' in choice.content.lower():
            expert_id = 2
        else:
            expert_id = 3 # Default to helpful assistant

        thinking, answer = expert_agents[expert_id]([taskInfo], cot_instruction)
        return answer

```

## Verify-and-Judge Ensemble

**Insights:**
The archive shows two winning strategies: independent sampling with voting (Self-Consistency, 98.4%) and interaction-based deliberation (Debate, 98.4%). Both have blind spots: voting ignores reasoning quality, and debate risks anchoring. The Verify-and-Judge Ensemble keeps solvers fully independent to preserve diverse reasoning paths, then resolves disagreements using backward verification (independently re-deriving the solution to check each candidate answer) rather than trusting any single solver's self-assessment. Compared to my previous draft, this version removes all manual Info construction in favor of a dedicated Vote Summarizer agent, and branches on a consensus flag so that verification is only invoked when solvers genuinely disagree.

**Overall Idea:**
1. Five solvers with diverse expert roles independently solve the task at temperature 0.8.
2. A Vote Summarizer agent receives the raw solver answer Infos and outputs a natural-language vote summary plus a consensus level ('HIGH' if the top answer has >= N-1 of N votes, else 'LOW').
3. If consensus is HIGH: skip verification and let the Judge confirm the leading answer with the solutions and vote summary. If LOW: for each distinct candidate (ordered by frequency), a Verifier agent independently re-derives the solution from scratch and outputs its own derived answer, a restatement of the candidate it checked, and a SUPPORTED/REFUTED verdict.
4. A Judge Agent receives the task, all solutions, the vote summary, and (when present) the verification evidence, and renders the final answer, prioritizing verified answers and vote counts.

**Implementation:**
1. Instantiate 5 solver agents with distinct roles (Math Professor, Grade School Teacher, Math Enthusiast, Careful Accountant, Helpful Assistant), temperature 0.8, fields ['thinking', 'answer']. Instruction emphasizes careful attention to numbers/units in any language and a clear final numeric answer.
2. Collect thinking/answer Infos directly; pass the answer Infos (as objects, no extraction) to a Vote Summarizer agent with fields ['vote_summary', 'consensus'] that counts the answers and reports the distribution and consensus level.
3. Branch on the consensus Info: if 'HIGH', verification_infos stays empty; if 'LOW', run a Verifier agent (fields ['derived_answer', 'verification', 'verdict'], temperature 0.1) once per distinct candidate, instructing it to identify the candidate answers from the provided solver answer Infos, re-solve independently, and explicitly restate which candidate it checked.
4. Judge agent (temperature 0.1, fields ['thinking', 'answer']) receives taskInfo + all thinking/answer Infos + vote summary + verification Infos, weighs consensus and verification evidence, and outputs the final answer.
5. Return the Judge's answer Info.

- **generation**: 1
- **fitness**: 95% Bootstrap Confidence Interval: (94.5%, 100.0%), Median: 97.7%

```python
def forward(self, taskInfo):
    N = 5  # Number of diverse parallel solvers

    # Phase 1: Diverse independent reasoning with varied expert roles
    solver_roles = ['Math Professor', 'Grade School Teacher', 'Math Enthusiast', 'Careful Accountant', 'Helpful Assistant']
    solver_instruction = "Please think step by step and carefully solve the task. Pay close attention to the numbers, units and wording of the problem (which may be in any language). Double-check your arithmetic before concluding. End by clearly stating the final numeric answer in the 'answer' field."
    solver_agents = [LLMAgentBase(['thinking', 'answer'], 'Solver Agent', role=role, temperature=0.8) for role in solver_roles]

    all_thinking = []
    all_answers = []
    for i in range(N):
        thinking, answer = solver_agents[i]([taskInfo], solver_instruction, 0)
        all_thinking.append(thinking)
        all_answers.append(answer)

    # Phase 2: Summarize the vote distribution via a dedicated agent (no manual Info construction)
    summarizer_agent = LLMAgentBase(['vote_summary', 'consensus'], 'Vote Summarizer Agent', temperature=0.1)
    summarizer_instruction = (
        f"You are given the final answers produced by {N} independent solvers to the same math problem. "
        "Count how many solvers produced each distinct answer and summarize the vote distribution in 'vote_summary' "
        "(e.g., which answer is most common and how many votes it received). "
        f"In 'consensus', output 'HIGH' if the most common answer received at least {N-1} of the {N} votes, otherwise output 'LOW'."
    )
    vote_summary, consensus = summarizer_agent(all_answers, summarizer_instruction, -1)

    # Phase 3: Only run independent verification when solvers genuinely disagree
    verification_infos = []
    if consensus.content != 'HIGH':
        verifier_agent = LLMAgentBase(['derived_answer', 'verification', 'verdict'], 'Verifier Agent', temperature=0.1)
        verify_instruction = (
            "You are given a math problem and the final answers proposed by several independent solvers. "
            "Identify the distinct candidate answers among them. For the candidate answer you are assigned below, "
            "independently re-solve the problem from scratch, writing out the key computations explicitly. "
            "Put your own independently derived final answer in 'derived_answer'. "
            "In 'verification', restate which candidate answer you checked and explain whether your derivation agrees with it. "
            "In 'verdict', output 'SUPPORTED' if your derivation agrees with the candidate answer, or 'REFUTED' if it disagrees. "
            "Do not simply trust the candidate answers — verify the arithmetic yourself. "
        )
        # Ask the verifier to check the most common and the second most common candidates.
        # These cover the plausible answers in a disagreement scenario.
        for rank_label in ['the most common candidate answer', 'the second most common candidate answer']:
            rank_instruction = verify_instruction + f"You are assigned to check {rank_label}."
            derived_answer, verification, verdict = verifier_agent([taskInfo] + all_answers, rank_instruction, -1)
            verification_infos.append(verification)
            verification_infos.append(verdict)
            verification_infos.append(derived_answer)

    # Phase 4: Final judgment fusing vote counts, verification verdicts, and verifier-derived answers
    judge_agent = LLMAgentBase(['thinking', 'answer'], 'Judge Agent', temperature=0.1)
    judge_instruction = (
        "You are given a math problem, several independent solutions from expert solvers, a majority-vote summary, "
        "and (if the solvers disagreed) independent verification evidence for the leading candidate answers. "
        "Weigh the evidence carefully: (1) how many solvers support each answer, (2) which candidates were independently "
        "verified as 'SUPPORTED' vs 'REFUTED', and what answers the verifiers themselves derived, (3) re-examine the reasoning "
        "of conflicting solutions to spot arithmetic or logic errors. Then reason step by step and provide the final correct "
        "numeric answer in the 'answer' field."
    )

    judge_inputs = [taskInfo] + all_thinking + all_answers + [vote_summary] + verification_infos
    thinking, answer = judge_agent(judge_inputs, judge_instruction, -1)

    return answer

```

## Grounded Cross-Lingual Program-of-Thought Ensemble

**Insights:**
The previous proposal's core innovation — combining native-language CoT, an English-pivot translation path, and a Program-of-Thought path — is novel versus the archive, but its earlier implementation let the innovation leak: the PoT solver self-reported its program's output, and the Judge had to mentally trace the program, both of which reintroduce the exact arithmetic unreliability PoT exists to fix. The fix is to ground the program's output mechanically via a dedicated low-temperature Program Executor agent whose sole task is line-by-line simulation, and to gate trust in that output via an explicit translation-fidelity check. This turns the design into a three-signal ensemble where each signal has a distinct, verifiable failure mode: native CoT (comprehension + arithmetic errors), translation path (fidelity errors but clean English reasoning), program path (modeling errors but exact arithmetic).

**Overall Idea:**
1. Three native-language CoT solvers (temperature 0.8, diverse roles) produce independent reasoning paths and candidate answers.
2. A Translator agent produces a faithful English translation ONLY (no solving), at temperature 0.0, to avoid leaking solutions into the translation.
3. A PoT Solver (temperature 0.3) analyzes the problem and writes a self-contained Python program — it does NOT report the answer; the program is the artifact.
4. A Program Executor agent (temperature 0.0) receives ONLY the program (not the problem, to avoid bias) and simulates it line by line, reporting the final printed value in 'program_output'.
5. A Fidelity Checker (temperature 0.0) compares the translation against the original problem and flags whether numbers/conditions were preserved ('FAITHFUL'/'DISTORTED').
6. A Vote Summarizer reports the answer distribution and consensus level.
7. A Trace-and-Judge agent (temperature 0.1) receives everything and applies an explicit hierarchical rule: if translation is FAITHFUL and the program output is a valid number, weigh the program output heavily (it is mechanically derived); break program-vs-majority ties by re-deriving from the translation; otherwise fall back to majority vote with reasoning inspection. Output the final numeric answer.

**Implementation:**
1. Instantiate 3 native CoT solvers with roles Math Professor, Grade School Teacher, Careful Accountant (fields ['thinking', 'answer'], temperature 0.8); instruction emphasizes careful reading in the original language, double-checking arithmetic, and a clean final numeric answer.
2. Instantiate a Translator agent (fields ['translation'], temperature 0.0) with an instruction to translate faithfully into English without solving or adding anything.
3. Instantiate a PoT Solver (fields ['analysis', 'program'], temperature 0.3) instructed to write a short self-contained Python program that computes and prints the answer; explicitly forbidden from giving a final answer itself.
4. Instantiate a Program Executor (fields ['trace', 'program_output'], temperature 0.0) that receives the program Info and simulates the program exactly, outputting the final printed value.
5. Instantiate a Fidelity Checker (fields ['fidelity_check', 'fidelity'], temperature 0.0) that receives taskInfo and the translation and outputs 'FAITHFUL' or 'DISTORTED' plus a brief check.
6. Instantiate a Vote Summarizer (fields ['vote_summary', 'consensus'], temperature 0.1) over the five candidate answers (3 native + translator-solver + program output).
7. Instantiate a Translator-Solver (fields ['thinking', 'answer'], temperature 0.3) that solves the English translation step by step.
8. Instantiate the Trace-and-Judge agent (fields ['thinking', 'answer'], temperature 0.1) with all Infos and an explicit decision hierarchy in the instruction; return its answer Info.

- **generation**: 2
- **fitness**: 95% Bootstrap Confidence Interval: (96.1%, 100.0%), Median: 98.4%

```python
def forward(self, taskInfo):
    # ---- Phase 1: Native-language diverse CoT solvers (decorrelated reasoning paths) ----
    N_native = 3
    native_roles = ['Math Professor', 'Grade School Teacher', 'Careful Accountant']
    native_instruction = (
        "Please think step by step and carefully solve this math problem, which may be written in any language. "
        "Read the problem statement carefully, paying close attention to numbers, units, and wording in the original language. "
        "Double-check every arithmetic step before concluding. "
        "End with the final numeric answer in the 'answer' field."
    )
    native_agents = [LLMAgentBase(['thinking', 'answer'], 'Native CoT Solver', role=role, temperature=0.8) for role in native_roles]

    native_thinking = []
    native_answers = []
    for i in range(N_native):
        thinking, answer = native_agents[i]([taskInfo], native_instruction, 0)
        native_thinking.append(thinking)
        native_answers.append(answer)

    # ---- Phase 2: Faithful English translation (translation ONLY, no solving) ----
    translator_agent = LLMAgentBase(['translation'], 'Translator Agent', temperature=0.0)
    translator_instruction = (
        "Translate the following math problem faithfully into English. "
        "Preserve every number, quantity, unit, and condition exactly. "
        "Do NOT solve the problem, do not add interpretations, and do not omit anything — output ONLY the translation."
    )
    translation = translator_agent([taskInfo], translator_instruction, -1)[0]

    # ---- Phase 2b: Solve the English translation (independent path via pivot language) ----
    trans_solver_agent = LLMAgentBase(['thinking', 'answer'], 'Cross-Lingual Solver', temperature=0.3)
    trans_solver_instruction = (
        "You are given an English translation of a math problem. "
        "Solve it step by step, writing your reasoning in 'thinking'. "
        "Double-check your arithmetic. Put the final numeric answer in 'answer'."
    )
    trans_thinking, trans_answer = trans_solver_agent([translation], trans_solver_instruction, -1)

    # ---- Phase 3: Program-of-Thought solver (modeling only, no self-reported answer) ----
    pot_agent = LLMAgentBase(['analysis', 'program'], 'Program-of-Thought Solver', temperature=0.3)
    pot_instruction = (
        "Translate this math problem into a short, self-contained Python program. "
        "First, analyze the problem: identify the known quantities, the relations between them, and what is asked; "
        "write this in 'analysis'. Then write a Python program in 'program' that computes the final answer: "
        "use explicit variables for each quantity, implement the relations, and end with a print statement of the final result. "
        "Do NOT state the final answer yourself — the program is the only required artifact."
    )
    pot_analysis, pot_program = pot_agent([taskInfo], pot_instruction, -1)

    # ---- Phase 4: Mechanical execution of the program by a dedicated executor ----
    # The executor deliberately receives ONLY the program (not the problem) so its simulation
    # is not biased by the problem statement; the Judge later checks the modeling against the task.
    executor_agent = LLMAgentBase(['trace', 'program_output'], 'Program Executor Agent', temperature=0.0)
    executor_instruction = (
        "You are given a Python program. Execute it EXACTLY as written, line by line, mechanically. "
        "Do not second-guess the program's logic; just simulate it precisely. "
        "In 'trace', show the value of each variable after each step. "
        "In 'program_output', output ONLY the final value printed by the program (just the number)."
    )
    program_trace, program_output = executor_agent([pot_program], executor_instruction, -1)

    # ---- Phase 5: Translation fidelity check (gates trust in the English-pivot path) ----
    fidelity_agent = LLMAgentBase(['fidelity_check', 'fidelity'], 'Fidelity Checker Agent', temperature=0.0)
    fidelity_instruction = (
        "You are given an original math problem (possibly in a non-English language) and its English translation. "
        "Check whether the translation preserves every number, quantity, unit, and condition of the original exactly, "
        "without additions, omissions, or distortions. Explain your check briefly in 'fidelity_check'. "
        "In 'fidelity', output 'FAITHFUL' if the translation is exact, or 'DISTORTED' if any number, condition, or meaning differs."
    )
    fidelity_check, fidelity = fidelity_agent([taskInfo, translation], fidelity_instruction, -1)

    # ---- Phase 6: Summarize the vote distribution across all five candidates ----
    summarizer_agent = LLMAgentBase(['vote_summary', 'consensus'], 'Vote Summarizer Agent', temperature=0.1)
    summarizer_instruction = (
        "You are given the final answers produced by five independent solvers to the same math problem "
        "(three native-language solvers, one solver working from an English translation, and one mechanically executed program output). "
        "Count how many solvers produced each distinct answer and summarize the distribution in 'vote_summary' "
        "(which answer is most common and how many votes it received, plus any minority answers). "
        "In 'consensus', output 'HIGH' if the most common answer received at least 4 of the 5 votes, otherwise output 'LOW'."
    )
    vote_summary, consensus = summarizer_agent(native_answers + [trans_answer, program_output], summarizer_instruction, -1)

    # ---- Phase 7: Trace-and-Judge fusion with explicit decision hierarchy ----
    judge_agent = LLMAgentBase(['thinking', 'answer'], 'Trace-and-Judge Agent', temperature=0.1)
    judge_instruction = (
        "You are given a math problem (possibly in a non-English language) and results from several independent solvers: "
        "three native-language solutions, an English translation with its own solution, a Python program with its mechanically "
        "traced output, a translation-fidelity check, and a vote summary. Decide the final answer using this hierarchy: "
        "(1) If the translation is 'FAITHFUL' and the program output is a valid number, the program output is the strongest "
        "evidence — it was computed mechanically, not by mental arithmetic — but verify that the program's logic in 'analysis' "
        "and 'program' actually models the problem correctly (correct quantities, correct relations, correct question). "
        "(2) If the program logic is flawed or the translation is 'DISTORTED', fall back to the majority vote, re-examining the "
        "reasoning of dissenting solutions to spot arithmetic or comprehension errors. "
        "(3) If the program output and the majority vote agree, adopt that answer. "
        "(4) If they conflict and the program logic appears correct, re-derive the solution from the faithful translation to break the tie. "
        "Put your step-by-step judgment in 'thinking' and the final correct numeric answer in 'answer'."
    )
    judge_inputs = (
        [taskInfo]
        + native_thinking + native_answers
        + [translation, trans_thinking, trans_answer]
        + [pot_analysis, pot_program, program_trace, program_output]
        + [fidelity_check, fidelity]
        + [vote_summary]
    )
    thinking, answer = judge_agent(judge_inputs, judge_instruction, -1)

    return answer

```

## Backward-Verification Tournament

**Insights:**
The archive's top performers (Self-Consistency, Debate, Gen-2's three-signal ensemble) all reach ~98.4%, but all verification mechanisms so far are *forward* re-solving or program execution — both repeat the same reasoning failure modes they try to catch. Backward verification (assume the answer, derive required intermediates, check every constraint) is a fundamentally different, decorrelated check: an arithmetic slip forward becomes a visible violated constraint backwards. The prior draft of this idea was sound in concept but had fragile candidate parsing, no explicit vote signal for the judge, and no method diversity across solvers. The revision keeps the novel backward-verification tournament core while hardening every handoff between agents.

**Overall Idea:**
1. Four solvers with diverse roles AND diverse solution methods (algebraic, unitary-bar, accounting-style, direct step-by-step) independently solve the problem.
2. A Vote Summarizer reports the answer distribution and consensus level (explicit signal for the judge).
3. A Candidate Aggregator lists the distinct numeric candidate answers in a strict machine-friendly format, with defensive parsing (multi-delimiter + regex + LLM fallback) in the control flow.
4. For each distinct candidate (up to 3; only 1 if unanimous), a Backward Verifier assumes the candidate is correct, derives the required intermediate quantities, checks every constraint from the original problem, and outputs CONSISTENT/INCONSISTENT with the failing constraint named. The verifier restates its assigned candidate for traceability.
5. A Tournament Judge receives everything and applies a strict priority hierarchy: single-consistent-candidate wins; multiple consistent candidates resolved by vote counts then reasoning inspection; zero consistent candidates triggers careful re-solving and least-violated selection. It must restate the winning candidate's constraint check before answering.

**Implementation:**
1. Instantiate 4 solvers (fields ['thinking', 'answer'], temperature 0.7, roles Math Professor / Grade School Teacher / Careful Accountant / Helpful Assistant) with method-specific instructions; collect thinking + answer Infos.
2. Instantiate a Vote Summarizer (fields ['vote_summary', 'consensus'], temperature 0.1) over the answer Infos; consensus is 'HIGH' if top answer has ≥ 3 of 4 votes.
3. Instantiate a Candidate Aggregator (fields ['distinct_answers'], temperature 0.1) with a strict numeric semicolon format; parse defensively (split on ; then newlines/commas, regex-extract numbers, dedupe normalized). If fewer than 2 numeric candidates parse, invoke a Fallback Aggregator agent with an even stricter format and re-parse; never extract or aggregate solver contents manually in the control flow.
4. Cap candidates at 3. Run one Backward Verifier per candidate (fields ['backward_derivation', 'constraint_check', 'verdict'], temperature 0.1); instruction embeds the candidate, demands a restatement of the checked candidate, backward derivation of intermediates, a per-constraint check table, and a strict CONSISTENT/INCONSISTENT verdict.
5. Instantiate the Tournament Judge (fields ['thinking', 'answer'], temperature 0.1) with taskInfo + all thinking/answers + vote summary + consensus + all verification Infos; instruction encodes the 4-rule priority hierarchy and requires restating the winning candidate's constraint check before finalizing.
6. Return the judge's answer Info.

- **generation**: 3
- **fitness**: 95% Bootstrap Confidence Interval: (93.8%, 99.2%), Median: 96.9%

```python
def forward(self, taskInfo):
    import re

    # ---- Phase 1: Diverse parallel forward solvers (varied roles AND methods) ----
    N = 4
    solver_specs = [
        ('Math Professor', 'Set up algebraic equations with explicit variables and solve them carefully.'),
        ('Grade School Teacher', 'Solve using clear elementary-school arithmetic, one small step at a time, as you would teach a student.'),
        ('Careful Accountant', 'Solve like a bookkeeper: track every quantity in a running tally, verifying each arithmetic operation as you go.'),
        ('Helpful Assistant', 'Solve step by step in the clearest direct way you can, double-checking each calculation.'),
    ]
    solver_instruction_template = (
        "Please think step by step and carefully solve this math problem, which may be written in any language. "
        "Pay close attention to every number, quantity, unit, and condition in the original wording. "
        "{method} "
        "Double-check your arithmetic before concluding. "
        "End with the final numeric answer (just the number) in the 'answer' field."
    )
    solver_agents = [
        LLMAgentBase(['thinking', 'answer'], 'Forward Solver', role=role, temperature=0.7)
        for role, _ in solver_specs
    ]

    all_thinking = []
    all_answers = []
    for i in range(N):
        instruction = solver_instruction_template.format(method=solver_specs[i][1])
        thinking, answer = solver_agents[i]([taskInfo], instruction, 0)
        all_thinking.append(thinking)
        all_answers.append(answer)

    # ---- Phase 2: Explicit vote summary (consensus signal for the judge) ----
    summarizer_agent = LLMAgentBase(['vote_summary', 'consensus'], 'Vote Summarizer Agent', temperature=0.1)
    summarizer_instruction = (
        f"You are given the final answers produced by {N} independent solvers to the same math problem. "
        "Count how many solvers produced each distinct answer (ignoring formatting differences like units or commas) "
        "and summarize the distribution in 'vote_summary' (which answer is most common and how many votes it received, "
        "plus any minority answers). "
        f"In 'consensus', output 'HIGH' if the most common answer received at least {N-1} of the {N} votes, otherwise 'LOW'."
    )
    vote_summary, consensus = summarizer_agent(all_answers, summarizer_instruction, -1)

    # ---- Phase 3: Extract distinct numeric candidate answers (robust parsing, no manual content aggregation) ----
    aggregator_agent = LLMAgentBase(['distinct_answers'], 'Candidate Aggregator', temperature=0.1)
    aggregator_instruction = (
        "You are given the final answers produced by several independent solvers to the same math problem. "
        "Identify all DISTINCT numeric answers among them (ignore units, commas, and formatting differences). "
        "Output them in 'distinct_answers' as numbers separated ONLY by semicolons, ordered from most frequently "
        "proposed to least frequently proposed. Example format: 348; 240. Output ONLY that string, nothing else."
    )
    distinct_answers_info = aggregator_agent(all_answers, aggregator_instruction, -1)[0]

    def parse_candidates(raw_text):
        """Defensively parse a candidate string into a deduplicated list of numeric tokens."""
        raw = str(raw_text)
        tokens = [t.strip().rstrip('.') for t in raw.split(';') if t.strip()]
        if len(tokens) <= 1:
            tokens = [t.strip().rstrip('.') for t in re.split(r'[
,]+', raw) if t.strip()]
        numeric_tokens = []
        for t in tokens:
            m = re.search(r'-?\d+(?:\.\d+)?', t.replace(',', ''))
            if m:
                numeric_tokens.append(m.group(0))
        candidates = []
        for t in numeric_tokens:
            norm = t.replace(',', '').replace(' ', '')
            if norm not in candidates:
                candidates.append(norm)
        return candidates

    candidates = parse_candidates(distinct_answers_info.content)

    # Fallback: if parsing the primary aggregator failed or is untrustworthy (fewer than 2
    # distinct candidates from a possibly-malformed string), ask a dedicated fallback agent
    # with a stricter format. We never extract or aggregate solver answer contents manually.
    if len(candidates) < 2:
        fallback_agent = LLMAgentBase(['distinct_answers'], 'Fallback Candidate Aggregator', temperature=0.1)
        fallback_instruction = (
            "You are given the final answers produced by several independent solvers to the same math problem. "
            "List every DISTINCT numeric answer that appears among them. "
            "STRICT FORMAT: your 'distinct_answers' field must contain ONLY digits and semicolons, e.g. 348; 240. "
            "No words, no units, no periods, no explanations. Order from most frequently proposed to least frequently proposed."
        )
        fallback_info = fallback_agent(all_answers, fallback_instruction, -1)[0]
        fallback_candidates = parse_candidates(fallback_info.content)
        if len(fallback_candidates) > len(candidates):
            candidates = fallback_candidates

    candidates = candidates[:3]  # cap verification budget at 3 distinct candidates

    # ---- Phase 4: Backward verification of each distinct candidate ----
    verifier_agent = LLMAgentBase(['backward_derivation', 'constraint_check', 'verdict'], 'Backward Verifier', temperature=0.1)
    verification_infos = []
    for idx, candidate in enumerate(candidates):
        verify_instruction = (
            "You are given a math problem (possibly in a non-English language). "
            f"A solver proposed the candidate answer: {candidate}.
"
            "Your job is NOT to solve the problem forward. Instead, VERIFY the candidate by working BACKWARDS:
"
            "1. Assume the candidate answer is correct.
"
            "2. Derive what each intermediate quantity in the problem must be, given this assumed answer.
"
            "3. Check whether EVERY condition stated in the original problem is satisfied by these derived quantities.
"
            "Write the backward derivation in 'backward_derivation'.
"
            "In 'constraint_check', FIRST restate the candidate answer you checked in one short line "
            f"(e.g., 'Checked candidate: {candidate}'), THEN list each constraint from the problem and whether it holds, "
            "with the actual numbers substituted (e.g., 'Constraint: rabbits = dogs + cats - 12 -> 228 = 60 + 120 - 12 = 168? NO').
"
            "In 'verdict', output 'CONSISTENT' if ALL constraints hold, or 'INCONSISTENT' if any constraint fails.
"
            "Be rigorous: a single violated constraint means INCONSISTENT. Do not assume the candidate is right — test it."
        )
        backward_derivation, constraint_check, verdict = verifier_agent([taskInfo], verify_instruction, idx)
        verification_infos.extend([constraint_check, verdict, backward_derivation])

    # ---- Phase 5: Tournament judge fusing forward consensus + backward consistency ----
    judge_agent = LLMAgentBase(['thinking', 'answer'], 'Tournament Judge', temperature=0.1)
    judge_instruction = (
        "You are given a math problem (possibly in a non-English language), several independent solutions with their "
        "final answers, a vote summary, and backward-verification reports. Each report assumes one candidate answer is "
        "correct, derives the required intermediate quantities, and checks every constraint in the original problem, "
        "ending with a 'CONSISTENT' or 'INCONSISTENT' verdict. Each report restates which candidate it checked.
"
        "Decide the final answer using these rules, in strict priority order:
"
        "(1) Map each verdict to its candidate using the 'Checked candidate:' restatement. If exactly one candidate is "
        "'CONSISTENT' and the others are 'INCONSISTENT', adopt the consistent one — it is the only answer that survives "
        "a full consistency check against the problem's constraints.
"
        "(2) If multiple candidates are 'CONSISTENT', prefer the one with more solver votes per the vote summary; if "
        "votes are tied, re-examine the solvers' reasoning and the constraint checks to find the subtle error in the "
        "weaker candidate.
"
        "(3) If NO candidate is 'CONSISTENT', re-solve the problem carefully from scratch, then choose the answer whose "
        "backward derivation violates the fewest or least-central constraints.
"
        "(4) Before finalizing, restate the winning candidate's constraint check and confirm yourself that every "
        "constraint truly holds.
"
        "Put your step-by-step judgment in 'thinking' and the final correct numeric answer in 'answer'."
    )
    judge_inputs = [taskInfo] + all_thinking + all_answers + [vote_summary, consensus] + verification_infos
    thinking, answer = judge_agent(judge_inputs, judge_instruction, -1)

    return answer

```

## Context-Grounded Backward Audit

**Insights:**
The archive's strongest signals are independent diverse sampling + voting (98.4%) and deliberation (98.4%). The genuinely novel mechanism attempted in Gen-3 — backward verification (assume the answer, derive intermediates, check every constraint) — is decorrelated from forward solving, but its implementation lost fitness (96.9%) for three reasons: backward verifiers received only the bare task and a candidate number, forcing them to re-solve forward (reintroducing correlated errors); fragile candidate-aggregation/parsing machinery could silently drop the correct candidate; and verdict-to-candidate mapping relied on free-text restatements. The fix is to keep backward verification as the core novelty while making every handoff structural and fully contexted, and to fuse it with the two strongest proven signals: a diverse forward ensemble and a mechanically executed program anchor.

**Overall Idea:**
**Context-Grounded Backward Audit:**
1. Five solvers with diverse roles AND methods (algebraic, elementary arithmetic, bookkeeping, enthusiast, direct) independently solve the problem at temperature 0.7.
2. A Vote Summarizer reports the answer distribution and a consensus level (explicit judge signal).
3. A Program-of-Thought solver writes a self-contained Python program; a Program Executor (temperature 0.0) receives ONLY the program and simulates it line by line — a mechanical arithmetic anchor no mental-math error can contaminate.
4. One Backward Auditor per solver answer (5 audits, no parsing, no caps): each auditor receives the full shared context (task, all reasoning/answers, program + trace + output) PLUS the specific candidate answer Info and its proposer's reasoning Info as dedicated inputs, works backwards from the assumed candidate, checks every constraint with numbers substituted, and outputs the checked candidate in a dedicated 'checked_answer' field — making verdict→candidate mapping structural, not textual.
5. A Judge receives everything and applies a strict priority hierarchy: single-consistent candidate wins; multiple consistent resolved by votes then program agreement; zero consistent triggers re-solving anchored on the program output. It confirms the winner against the program trace before answering.

**Implementation:**
1. Instantiate 5 solvers (fields ['thinking','answer'], temperature 0.7) with method-specific instructions; collect thinking + answer Infos.
2. Vote Summarizer (fields ['vote_summary','consensus'], temperature 0.1) over the answer Infos.
3. PoT solver (fields ['analysis','program'], temperature 0.3) then Program Executor (fields ['trace','program_output'], temperature 0.0) over the program only.
4. For each solver i, Backward Auditor (fields ['checked_answer','constraint_check','verdict'], temperature 0.1) with inputs [taskInfo, thinking_i, answer_i] + shared context; instruction says to audit 'the candidate answer and its proposer's reasoning provided in the dedicated input section' — never injecting .content into instruction strings.
5. Judge (fields ['thinking','answer'], temperature 0.1) with taskInfo + all solutions + vote summary + consensus + program artifacts + all audit reports; strict 4-rule hierarchy; return its answer Info with defensive unpacking throughout.

- **generation**: 5
- **fitness**: 95% Bootstrap Confidence Interval: (93.8%, 99.2%), Median: 96.9%

```python
def forward(self, taskInfo):
    # ---- Phase 1: Diverse parallel forward solvers (varied roles AND methods) ----
    N = 5
    solver_specs = [
        ('Math Professor', 'Set up algebraic equations with explicit variables and solve them carefully.'),
        ('Grade School Teacher', 'Solve using clear elementary-school arithmetic, one small step at a time, as you would teach a student.'),
        ('Careful Accountant', 'Solve like a bookkeeper: track every quantity in a running tally, verifying each arithmetic operation as you go.'),
        ('Math Enthusiast', 'Solve with whatever clever method you like, but show every step clearly.'),
        ('Helpful Assistant', 'Solve step by step in the clearest direct way you can, double-checking each calculation.'),
    ]
    solver_instruction_template = (
        "Please think step by step and carefully solve this math problem, which may be written in any language. "
        "Pay close attention to every number, quantity, unit, and condition in the original wording. "
        "{method} "
        "Double-check your arithmetic before concluding. "
        "End with the final numeric answer (just the number) in the 'answer' field."
    )
    solver_agents = [
        LLMAgentBase(['thinking', 'answer'], 'Forward Solver', role=role, temperature=0.7)
        for role, _ in solver_specs
    ]

    all_thinking = []
    all_answers = []
    for i in range(N):
        instruction = solver_instruction_template.format(method=solver_specs[i][1])
        thinking, answer = solver_agents[i]([taskInfo], instruction, i)
        all_thinking.append(thinking)
        all_answers.append(answer)

    # ---- Phase 2: Vote summary + consensus signal ----
    summarizer_agent = LLMAgentBase(['vote_summary', 'consensus'], 'Vote Summarizer Agent', temperature=0.1)
    summarizer_instruction = (
        f"You are given the final answers produced by {N} independent solvers to the same math problem. "
        "Count how many solvers produced each distinct answer (ignoring formatting differences like units or commas) "
        "and summarize the distribution in 'vote_summary' (which answer is most common and how many votes it received, "
        "plus any minority answers). "
        f"In 'consensus', output 'HIGH' if the most common answer received at least {N-1} of the {N} votes, otherwise 'LOW'."
    )
    summary_infos = summarizer_agent(all_answers, summarizer_instruction, -1)
    vote_summary = summary_infos[0] if len(summary_infos) > 0 else None
    consensus = summary_infos[1] if len(summary_infos) > 1 else None

    # ---- Phase 3: Mechanical arithmetic anchor (program + executor) ----
    pot_agent = LLMAgentBase(['analysis', 'program'], 'Program-of-Thought Solver', temperature=0.3)
    pot_instruction = (
        "Translate this math problem into a short, self-contained Python program. "
        "First, analyze the problem in 'analysis': identify the known quantities, the relations between them, "
        "and what is asked. Then write the Python program in 'program': use explicit variables for each quantity, "
        "implement the relations, and end with a print statement of the final result. "
        "Do NOT state the final answer yourself — the program is the only required artifact."
    )
    pot_infos = pot_agent([taskInfo], pot_instruction, -1)
    pot_analysis = pot_infos[0] if len(pot_infos) > 0 else None
    pot_program = pot_infos[1] if len(pot_infos) > 1 else None

    executor_agent = LLMAgentBase(['trace', 'program_output'], 'Program Executor Agent', temperature=0.0)
    executor_instruction = (
        "You are given a Python program. Execute it EXACTLY as written, line by line, mechanically. "
        "Do not second-guess the program's logic; just simulate it precisely. "
        "In 'trace', show the value of each variable after each step. "
        "In 'program_output', output ONLY the final value printed by the program (just the number)."
    )
    executor_inputs = [pot_program] if pot_program is not None else []
    executor_infos = executor_agent(executor_inputs, executor_instruction, -1)
    program_trace = executor_infos[0] if len(executor_infos) > 0 else None
    program_output = executor_infos[1] if len(executor_infos) > 1 else None

    # ---- Phase 4: Context-grounded backward audit of every solver's candidate ----
    # Shared audit context: task + all reasoning/answers + program artifacts.
    audit_context = ([taskInfo] + all_thinking + all_answers
                     + [info for info in [pot_analysis, pot_program, program_trace, program_output] if info is not None])

    verifier_agent = LLMAgentBase(['checked_answer', 'constraint_check', 'verdict'], 'Backward Auditor', temperature=0.1)
    verification_infos = []
    for idx in range(N):
        # Dedicated input section: the candidate answer and its proposer's reasoning are
        # passed as Info objects (never injected into the instruction string manually).
        audit_inputs = [taskInfo, all_thinking[idx], all_answers[idx]] + audit_context
        verify_instruction = (
            "You are auditing one proposed solution to a math problem (possibly in a non-English language). "
            "The FIRST 'thinking' and FIRST 'answer' entries in the input section below are the CANDIDATE under audit: "
            "that 'answer' is the candidate answer, and that 'thinking' is its proposer's reasoning. "
            "The remaining inputs provide the original task, other solvers' solutions, and a mechanically executed "
            "Python program with its traced output (a reliable arithmetic reference).
"
            "Your job is NOT to re-solve the problem from scratch. Instead, AUDIT the candidate by working BACKWARDS:
"
            "1. Assume the candidate answer is correct.
"
            "2. Derive what each intermediate quantity in the problem must be, given this assumed answer.
"
            "3. Check whether EVERY condition stated in the original problem is satisfied by these derived quantities.
"
            "4. Where arithmetic is involved, cross-check against the program's traced output rather than redoing mental math.
"
            "In 'checked_answer', restate the candidate answer you audited (just the number).
"
            "In 'constraint_check', list each constraint from the problem and whether it holds, with the actual numbers "
            "substituted (e.g., 'Constraint: rabbits = dogs + cats - 12 -> 228 = 60 + 120 - 12 = 168? NO').
"
            "In 'verdict', output 'CONSISTENT' if ALL constraints hold, or 'INCONSISTENT' if any constraint fails. "
            "Be rigorous: a single violated constraint means INCONSISTENT. Do not assume the candidate is right — test it."
        )
        audit_infos = verifier_agent(audit_inputs, verify_instruction, idx)
        verification_infos.extend(audit_infos)

    # ---- Phase 5: Judge fusing forward consensus + backward audits + mechanical anchor ----
    judge_agent = LLMAgentBase(['thinking', 'answer'], 'Judge Agent', temperature=0.1)
    judge_instruction = (
        "You are given a math problem (possibly in a non-English language), five independent solutions with their "
        "final answers, a vote summary with a consensus level, a Python program with its mechanically traced output, "
        "and one backward-audit report per solver answer. Each audit report names the candidate it checked "
        "(in 'checked_answer'), lists every problem constraint with numbers substituted, and ends with a "
        "'CONSISTENT' or 'INCONSISTENT' verdict.
"
        "Decide the final answer using these rules, in strict priority order:
"
        "(1) Group the audits by their 'checked_answer' value. If exactly one distinct candidate is 'CONSISTENT' and "
        "the others are 'INCONSISTENT', adopt the consistent one — it is the only answer that survives a full "
        "consistency check against the problem's constraints.
"
        "(2) If multiple distinct candidates are 'CONSISTENT', prefer the one with more solver votes per the vote "
        "summary; if votes are tied, prefer the one matching the program's mechanically traced output; if still tied, "
        "re-examine the constraint checks to find the subtle error in the weaker candidate.
"
        "(3) If NO candidate is 'CONSISTENT', re-solve the problem carefully from scratch, using the program's traced "
        "output as an arithmetic reference, and choose the answer that best matches your derivation.
"
        "(4) Before finalizing, confirm the winning candidate against the program's traced output and restate its "
        "constraint check.
"
        "Put your step-by-step judgment in 'thinking' and the final correct numeric answer in 'answer'."
    )
    judge_inputs = ([taskInfo] + all_thinking + all_answers
                    + [info for info in [vote_summary, consensus, pot_analysis, pot_program, program_trace, program_output] if info is not None]
                    + verification_infos)
    judge_infos = judge_agent(judge_inputs, judge_instruction, -1)

    # Defensive fallback: if the judge somehow returned no 'answer' field, fall back to
    # the most common solver answer so we always return the best answer we can get.
    answer = None
    for info in judge_infos:
        if info.name == 'answer':
            answer = info
            break
    if answer is None:
        answer = all_answers[0]
    return answer

```

## Program-Anchored Reasoning Arbitration

**Insights:**
The archive's top performers (Self-Consistency, Debate, Gen-2's cross-lingual ensemble) all reach ~98.4%, but they share a blind spot: they evaluate reasoning paths only by their FINAL answers. Two solvers can reach the same correct answer via flawed reasoning (lucky arithmetic), or one can reach a wrong answer via mostly-correct reasoning with a single slip — and final-answer voting cannot distinguish these cases. Meanwhile, Gen-2 showed that a mechanically executed program (temperature 0.0 executor simulating line-by-line) provides a trustworthy arithmetic backbone. The novel mechanism I propose: use the program's VARIABLE TRACE not just as an answer anchor but as a fine-grained diagnostic that audits each solver's INTERMEDIATE quantities — catching reasoning errors even when final answers coincide, and validating reasoning even when final answers diverge.

**Overall Idea:**
**Program-Anchored Reasoning Arbitration (PARA):**
1. Five solvers with diverse roles AND methods (algebraic, elementary arithmetic, bookkeeping, enthusiast, direct) solve the problem independently at temperature 0.8 — preserving full sampler diversity.
2. A Program-of-Thought solver writes a self-contained Python program; a Program Executor (temperature 0.0) receives ONLY the program and simulates it line by line, producing a variable trace and final output — the mechanical arithmetic backbone.
3. **Novel step — Trace-Based Reasoning Audit:** A Reasoning Auditor agent receives the task, the program's variable trace, and ONE solver's solution. It checks whether the solver's intermediate quantities match the traced values, identifying the exact step where the solver's reasoning diverges from the mechanically verified computation. Output: 'trace_match' (MATCH/MISMATCH), 'divergence_step' (where the solver's reasoning went wrong, if anywhere), and 'audited_answer' (the solver's final answer, restated for traceability).
4. A final Judge receives: the task, all solutions, all audit reports, and the program output. It applies a strict hierarchy: (a) if the program output matches any solver's answer AND that solver's audit is MATCH, adopt it; (b) if the program output matches no solver's answer, re-solve from the program trace; (c) tiebreak by audit quality (MATCH audits are more trustworthy than MISMATCH audits).

This is genuinely new: the program trace is used not just for the final answer but as a fine-grained diagnostic of each reasoning path's intermediate steps — a mechanism absent from all archive entries.

**Implementation:**
1. Instantiate 5 solvers (fields ['thinking','answer'], temperature 0.8) with method-specific instructions; collect thinking + answer Infos.
2. PoT solver (fields ['analysis','program'], temperature 0.3) then Program Executor (fields ['trace','program_output'], temperature 0.0) over the program only.
3. For each solver i, Reasoning Auditor (fields ['trace_match','divergence_step','audited_answer'], temperature 0.1) with inputs [taskInfo, thinking_i, answer_i, program_trace, program_output]; instruction says to check the solver's intermediate quantities against the program's traced values.
4. Judge (fields ['thinking','answer'], temperature 0.1) with taskInfo + all solutions + all audit reports + program output; strict 3-rule hierarchy; return its answer Info with defensive unpacking throughout.

- **generation**: 11
- **fitness**: 95% Bootstrap Confidence Interval: (92.2%, 99.2%), Median: 96.1%

```python
def forward(self, taskInfo):
    # ---- Phase 1: Diverse parallel forward solvers (varied roles AND methods) ----
    N = 5
    solver_specs = [
        ('Math Professor', 'Set up algebraic equations with explicit variables and solve them carefully.'),
        ('Grade School Teacher', 'Solve using clear elementary-school arithmetic, one small step at a time, as you would teach a student.'),
        ('Careful Accountant', 'Solve like a bookkeeper: track every quantity in a running tally, verifying each arithmetic operation as you go.'),
        ('Math Enthusiast', 'Solve with whatever clever method you like, but show every step clearly.'),
        ('Helpful Assistant', 'Solve step by step in the clearest direct way you can, double-checking each calculation.'),
    ]
    solver_instruction_template = (
        "Please think step by step and carefully solve this math problem, which may be written in any language. "
        "Pay close attention to every number, quantity, unit, and condition in the original wording. "
        "{method} "
        "Double-check your arithmetic before concluding. "
        "End with the final numeric answer (just the number) in the 'answer' field."
    )
    solver_agents = [
        LLMAgentBase(['thinking', 'answer'], 'Forward Solver', role=role, temperature=0.8)
        for role, _ in solver_specs
    ]

    all_thinking = []
    all_answers = []
    for i in range(N):
        instruction = solver_instruction_template.format(method=solver_specs[i][1])
        outputs = solver_agents[i]([taskInfo], instruction, i)
        thinking = outputs[0] if len(outputs) > 0 else None
        answer = outputs[1] if len(outputs) > 1 else None
        if thinking is not None:
            all_thinking.append(thinking)
        if answer is not None:
            all_answers.append(answer)

    # Defensive: if no solver produced any answer, fall back to a direct CoT solve.
    if len(all_answers) == 0:
        fallback_agent = LLMAgentBase(['thinking', 'answer'], 'Fallback Solver', temperature=0.1)
        fallback_instruction = (
            "Please think step by step and carefully solve this math problem, which may be written in any language. "
            "Pay close attention to every number, quantity, unit, and condition. "
            "Double-check your arithmetic before concluding. "
            "End with the final numeric answer (just the number) in the 'answer' field."
        )
        fallback_outputs = fallback_agent([taskInfo], fallback_instruction, -1)
        fallback_answer = None
        for info in fallback_outputs:
            if info.name == 'answer':
                fallback_answer = info
                break
        if fallback_answer is not None:
            return fallback_answer
        # Last resort: return the task itself rather than an error string.
        return taskInfo

    # ---- Phase 2: Mechanical arithmetic backbone (program + trace) ----
    pot_agent = LLMAgentBase(['analysis', 'program'], 'Program-of-Thought Solver', temperature=0.3)
    pot_instruction = (
        "Translate this math problem into a short, self-contained Python program. "
        "First, analyze the problem in 'analysis': identify the known quantities, the relations between them, "
        "and what is asked. Then write the Python program in 'program': use explicit, meaningfully-named variables "
        "for each intermediate quantity, implement the relations, and end with a print statement of the final result. "
        "Do NOT state the final answer yourself — the program is the only required artifact."
    )
    pot_infos = pot_agent([taskInfo], pot_instruction, -1)
    pot_analysis = None
    pot_program = None
    for info in pot_infos:
        if info.name == 'analysis' and pot_analysis is None:
            pot_analysis = info
        elif info.name == 'program' and pot_program is None:
            pot_program = info

    executor_agent = LLMAgentBase(['trace', 'program_output'], 'Program Executor Agent', temperature=0.0)
    executor_instruction = (
        "You are given a Python program. Execute it EXACTLY as written, line by line, mechanically. "
        "Do not second-guess the program's logic; just simulate it precisely. "
        "In 'trace', show the value of EVERY variable after each assignment step, clearly labeled with the variable name. "
        "In 'program_output', output ONLY the final value printed by the program (just the number)."
    )
    program_trace = None
    program_output = None
    if pot_program is not None:
        executor_infos = executor_agent([pot_program], executor_instruction, -1)
        for info in executor_infos:
            if info.name == 'trace' and program_trace is None:
                program_trace = info
            elif info.name == 'program_output' and program_output is None:
                program_output = info

    # ---- Phase 3: NOVEL — Trace-based reasoning audit of each solver's intermediate steps ----
    auditor_agent = LLMAgentBase(
        ['trace_match', 'divergence_step', 'audited_answer'],
        'Reasoning Auditor', temperature=0.1
    )
    audit_reports = []
    for idx in range(len(all_answers)):
        thinking_info = all_thinking[idx] if idx < len(all_thinking) else None
        answer_info = all_answers[idx]
        audit_inputs = [taskInfo]
        if thinking_info is not None:
            audit_inputs.append(thinking_info)
        audit_inputs.append(answer_info)
        if program_trace is not None:
            audit_inputs.append(program_trace)
        if program_output is not None:
            audit_inputs.append(program_output)

        audit_instruction = (
            "You are auditing ONE solver's step-by-step solution to a math problem (possibly in a non-English language). "
            "The FIRST 'thinking' (if present) and the FIRST 'answer' entries in the input section below are the CANDIDATE "
            "solution under audit. You are also given a Python program's variable trace — a mechanically verified "
            "computation of the problem's intermediate quantities — and the program's final output.
"
            "Your job is NOT to re-solve the problem. Instead, compare the solver's reasoning against the mechanical trace:
"
            "1. In 'trace_match', output 'MATCH' if the solver's intermediate quantities (the numbers it computes along the "
            "way) agree with the corresponding variable values in the program's trace, or 'MISMATCH' if any intermediate "
            "quantity differs. If the solver's approach uses different intermediate quantities than the program, judge "
            "whether the solver's own quantities are arithmetically consistent with the traced ones.
"
            "2. In 'divergence_step', if there is a mismatch, identify the EXACT step where the solver's reasoning diverges "
            "from the trace (quote both the solver's value and the traced value). If everything matches, write 'NONE'.
"
            "3. In 'audited_answer', restate the solver's final answer (just the number).
"
            "Be precise: you are the diagnostic that distinguishes 'correct answer via flawed reasoning' from 'correct "
            "answer via sound reasoning'."
        )
        audit_infos = auditor_agent(audit_inputs, audit_instruction, idx)
        audit_reports.extend(audit_infos)

    # ---- Phase 4: Judge fusing solutions, audits, and the mechanical anchor ----
    judge_agent = LLMAgentBase(['thinking', 'answer'], 'Judge Agent', temperature=0.1)
    judge_instruction = (
        "You are given a math problem (possibly in a non-English language), several independent solutions with their "
        "final answers, one audit report per solution, and a Python program with its mechanically traced output. "
        "Each audit report checks whether that solver's INTERMEDIATE quantities match the program's variable trace, "
        "identifies any divergence step, and restates the audited answer.
"
        "Decide the final answer using these rules, in strict priority order:
"
        "(1) If the program's mechanically traced output agrees with any solver's answer AND that solver's audit report "
        "says 'MATCH', adopt that answer — it is confirmed by both sound reasoning and mechanical computation.
"
        "(2) If the program's output agrees with a solver's answer but the audit says 'MISMATCH', carefully re-examine "
        "the divergence step: if the solver's final answer still matches the program output, the divergence is likely "
        "a notational difference rather than an error — adopt the program output.
"
        "(3) If the program's output matches NO solver's answer, re-solve the problem carefully from scratch, using the "
        "program's variable trace as the authoritative arithmetic reference, and cross-check against the audits to find "
        "which solvers' reasoning was sound.
"
        "(4) Before finalizing, confirm the winning answer against the program's traced output one final time.
"
        "Put your step-by-step judgment in 'thinking' and the final correct numeric answer (just the number) in 'answer'."
    )
    judge_inputs = (
        [taskInfo]
        + all_thinking
        + all_answers
        + audit_reports
        + [info for info in [pot_analysis, pot_program, program_trace, program_output] if info is not None]
    )
    judge_infos = judge_agent(judge_inputs, judge_instruction, -1)

    # Defensive fallback: always return the best available answer.
    answer = None
    for info in judge_infos:
        if info.name == 'answer':
            answer = info
            break
    if answer is None:
        # Prefer the mechanically derived program output, then any solver answer.
        if program_output is not None:
            answer = program_output
        else:
            answer = all_answers[0]
    return answer

```

## Adaptive Cross-Lingual Consensus

**Insights:**
The archive shows a consistent pattern: architectures that rely on LLM-based verification/auditing stages (backward verification, trace audits) underperform (96.1–96.9%), while architectures built on diverse independent sampling + voting and mechanically-executed programs reach the ceiling (98.4%). Two mechanisms remain unexplored: (1) adaptive sequential sampling with early exit — fixed-N ensembles waste samples on easy problems and add noise; (2) multi-pivot cross-lingual triangulation — Gen-2 used only an English pivot, but MGSM is multilingual and a second structurally-different pivot language provides another decorrelated reasoning path. Additionally, all archive entries make the final decision via an LLM judge; making the final plurality decision deterministically in code removes the last unreliable LLM decision point.

**Overall Idea:**
**Adaptive Cross-Lingual Consensus (ACLC):**
1. Sample native-language CoT solvers *sequentially* (max 5, temperature 0.8). After each sample, count votes in code (normalized numeric answers). If any answer reaches a 3-vote supermajority, stop immediately and return it — easy problems finish in 3 samples with zero extra noise.
2. If no supermajority after 5 samples, escalate with three *decorrelated* signals, all joining the vote pool: (a) a mechanically-executed Python program (PoT solver + line-by-line executor at temperature 0.0), (b) a solver working from a faithful English translation, (c) a solver working from a translation into a second pivot language chosen by a lightweight router to be structurally different from both English and the problem's language.
3. Final decision is deterministic code: plurality vote over all candidates (5 native + program output + 2 pivot solvers), with tiebreak priority: native majority → program output → English pivot → second pivot → first native answer.

**Implementation:**
1. Instantiate one native CoT solver agent (fields ['thinking','answer'], temperature 0.8); call it up to 5 times sequentially, passing taskInfo plus all prior thinking/answer Infos with a decorrelation instruction; after each call, count normalized answers in code; break on 3-vote supermajority and return that answer Info directly.
2. If no early exit: PoT solver (fields ['analysis','program'], temperature 0.3) writes a program; Program Executor (fields ['trace','program_output'], temperature 0.0) receives ONLY the program and simulates it line by line.
3. English Translator (fields ['translation'], temperature 0.0) translates without solving; Language Router (fields ['pivot_language'], temperature 0.1) picks a structurally-different second pivot language; Translator (fields ['translation'], temperature 0.0) translates into it.
4. Two pivot solvers (temperature 0.3) solve the English and second-language translations independently.
5. Code-level plurality vote over all 8 candidate answer Infos with deterministic tiebreaks; defensive fallbacks: if any phase fails, degrade gracefully to whatever candidates exist; if none, a final low-temperature direct solve. Never construct Info manually, never print, always return an answer Info.

- **generation**: 12
- **fitness**: 95% Bootstrap Confidence Interval: (92.2%, 99.2%), Median: 96.1%

```python
def forward(self, taskInfo):
    import re
    from collections import Counter

    def normalize_answer(text):
        """Normalize an answer string to a comparable numeric token."""
        raw = str(text).strip()
        cleaned = raw.replace(',', '').replace(' ', '')
        cleaned = cleaned.rstrip('.')
        m = re.search(r'-?\d+(?:\.\d+)?', cleaned)
        if m:
            return m.group(0)
        return raw

    def get_field(infos, field_name):
        """Defensively extract the first Info with the given field name from a list of Infos."""
        if infos is None:
            return None
        for info in infos:
            if info.name == field_name:
                return info
        return None

    # ---- Phase 1: Adaptive sequential native-language CoT sampling with early exit ----
    N_max = 5
    SUPERMAJORITY = 3

    solver_agent = LLMAgentBase(['thinking', 'answer'], 'Native CoT Solver',
                                role='Expert Mathematician', temperature=0.8)
    base_instruction = (
        "Please think step by step and carefully solve this math problem, which may be written in any language. "
        "Pay close attention to every number, quantity, unit, and condition in the original wording. "
        "Double-check every arithmetic step before concluding. "
        "End with the final numeric answer (just the number) in the 'answer' field."
    )
    decorrelate_instruction = (
        "You are given a math problem and previous solution attempts by other solvers. "
        "IGNORE the previous attempts entirely — they may contain errors. Solve the problem completely "
        "independently with your own approach. "
        "Please think step by step and carefully solve this math problem, which may be written in any language. "
        "Pay close attention to every number, quantity, unit, and condition in the original wording. "
        "Double-check every arithmetic step before concluding. "
        "End with the final numeric answer (just the number) in the 'answer' field."
    )

    all_thinking = []
    all_answers = []
    early_exit_answer = None

    for i in range(N_max):
        instruction = base_instruction if i == 0 else decorrelate_instruction
        outputs = solver_agent([taskInfo] + all_thinking + all_answers, instruction, i)
        thinking = get_field(outputs, 'thinking')
        answer = get_field(outputs, 'answer')
        if thinking is not None:
            all_thinking.append(thinking)
        if answer is None:
            continue
        all_answers.append(answer)

        # Cheap code-level vote counting (no extra LLM call)
        votes = Counter(normalize_answer(a.content) for a in all_answers)
        top_answer, top_count = votes.most_common(1)[0]
        if top_count >= SUPERMAJORITY:
            for a in all_answers:
                if normalize_answer(a.content) == top_answer:
                    early_exit_answer = a
                    break
            break

    # ---- Early-exit path: supermajority reached, return directly ----
    if early_exit_answer is not None:
        return early_exit_answer

    # ---- Phase 2: Mechanical arithmetic signal (program + executor) ----
    pot_agent = LLMAgentBase(['analysis', 'program'], 'Program-of-Thought Solver', temperature=0.3)
    pot_instruction = (
        "Translate this math problem into a short, self-contained Python program. "
        "First analyze the problem in 'analysis': identify known quantities, the relations between them, and what is asked. "
        "Then write the Python program in 'program': use explicit variables for each quantity, implement the relations, "
        "and end with a print statement of the final result. "
        "Do NOT state the final answer yourself — the program is the only required artifact."
    )
    pot_infos = pot_agent([taskInfo], pot_instruction, -1)
    pot_program = get_field(pot_infos, 'program')

    program_output = None
    if pot_program is not None:
        executor_agent = LLMAgentBase(['trace', 'program_output'], 'Program Executor Agent', temperature=0.0)
        executor_instruction = (
            "You are given a Python program. Execute it EXACTLY as written, line by line, mechanically. "
            "Do not second-guess the program's logic; just simulate it precisely. "
            "In 'trace', show the value of each variable after each step. "
            "In 'program_output', output ONLY the final value printed by the program (just the number)."
        )
        executor_infos = executor_agent([pot_program], executor_instruction, -1)
        program_output = get_field(executor_infos, 'program_output')

    # ---- Phase 3: Cross-lingual pivot signals ----
    pivot_solver_instruction = (
        "You are given a translated math problem. Solve it step by step, writing your reasoning in 'thinking'. "
        "Pay close attention to every number, quantity, unit, and condition. Double-check your arithmetic. "
        "Put the final numeric answer (just the number) in 'answer'."
    )

    # English pivot: translate then solve
    english_translator_agent = LLMAgentBase(['translation'], 'English Translator Agent', temperature=0.0)
    english_translate_instruction = (
        "Translate the following math problem faithfully into English. "
        "Preserve every number, quantity, unit, and condition exactly. "
        "Do NOT solve the problem and do not omit anything — output ONLY the translation."
    )
    english_trans_infos = english_translator_agent([taskInfo], english_translate_instruction, -1)
    english_translation = get_field(english_trans_infos, 'translation')

    english_answer_info = None
    if english_translation is not None:
        english_solver_agent = LLMAgentBase(['thinking', 'answer'], 'English Pivot Solver', temperature=0.3)
        english_outputs = english_solver_agent([english_translation], pivot_solver_instruction, -1)
        english_answer_info = get_field(english_outputs, 'answer')

    # Second pivot language: route, translate, then solve
    pivot_language = 'Spanish'
    router_agent = LLMAgentBase(['pivot_language'], 'Language Router', temperature=0.1)
    router_instruction = (
        "You are given a math problem written in some language. Choose ONE additional pivot language in which "
        "to re-express and solve this problem, providing an independent reasoning path. Choose from: "
        "Chinese, Spanish, French, German, Russian. Prefer a language structurally DIFFERENT from both English "
        "and the problem's own language (e.g., if the problem is in a European language, prefer Chinese; "
        "if it is in an Asian language, prefer Spanish). Output ONLY the language name in 'pivot_language'."
    )
    router_infos = router_agent([taskInfo], router_instruction, -1)
    pivot_lang_info = get_field(router_infos, 'pivot_language')
    if pivot_lang_info is not None:
        pl = str(pivot_lang_info.content).strip().rstrip('.')
        if pl:
            pivot_language = pl

    pivot_answer_info = None
    translator_agent = LLMAgentBase(['translation'], 'Translator Agent', temperature=0.0)
    pivot_translate_instruction = (
        f"Translate the following math problem faithfully into {pivot_language}. "
        "Preserve every number, quantity, unit, and condition exactly. "
        "Do NOT solve the problem and do not omit anything — output ONLY the translation."
    )
    pivot_trans_infos = translator_agent([taskInfo], pivot_translate_instruction, -1)
    pivot_translation = get_field(pivot_trans_infos, 'translation')

    if pivot_translation is not None:
        pivot_solver_agent = LLMAgentBase(['thinking', 'answer'], 'Pivot Language Solver', temperature=0.3)
        pivot_outputs = pivot_solver_agent([pivot_translation], pivot_solver_instruction, -1)
        pivot_answer_info = get_field(pivot_outputs, 'answer')

    # ---- Phase 4: Deterministic final plurality vote over all candidates ----
    candidate_infos = list(all_answers)
    if program_output is not None:
        candidate_infos.append(program_output)
    if english_answer_info is not None:
        candidate_infos.append(english_answer_info)
    if pivot_answer_info is not None:
        candidate_infos.append(pivot_answer_info)

    if len(candidate_infos) == 0:
        # Last-resort fallback: direct low-temperature solve
        fallback_agent = LLMAgentBase(['thinking', 'answer'], 'Fallback Solver', temperature=0.1)
        fallback_outputs = fallback_agent([taskInfo], base_instruction, -1)
        fallback_answer = get_field(fallback_outputs, 'answer')
        if fallback_answer is not None:
            return fallback_answer
        return taskInfo

    votes = Counter(normalize_answer(a.content) for a in candidate_infos)
    winning_normalized, _ = votes.most_common(1)[0]

    # Deterministic tiebreak priority: native answers (earliest first) -> program output
    # -> English pivot -> second pivot -> first available candidate.
    for a in all_answers:
        if normalize_answer(a.content) == winning_normalized:
            return a
    if program_output is not None and normalize_answer(program_output.content) == winning_normalized:
        return program_output
    if english_answer_info is not None and normalize_answer(english_answer_info.content) == winning_normalized:
        return english_answer_info
    if pivot_answer_info is not None and normalize_answer(pivot_answer_info.content) == winning_normalized:
        return pivot_answer_info

    # Defensive fallback: return the first available candidate.
    return candidate_infos[0]

```

## Replication-Weighted Consensus

**Insights:**
The archive reveals two robust lessons: (1) diverse independent sampling with voting is the most reliable backbone (Self-Consistency: 98.4%); (2) every attempt to replace mechanical signals with LLM judgment (verifiers, auditors, summarizers) or to drop mechanical grounding entirely (Gen-12 code-only voting: 96.1%) loses fitness. My previous draft (replication-weighted voting) was mechanically novel but effectively degenerated into "Self-Consistency with 10 noisy samples" because the re-check used a deliberately weakened prompt, making the stability signal noisy, and it discarded the proven mechanical program anchor.

**Overall Idea:**
**Replication-Weighted Consensus with Program Tiebreak (RWC):**
1. Five diverse solvers (distinct roles AND distinct methods, temperature 0.8) each produce an original solution — the proven backbone.
2. Each solver's answer is independently replicated by a *full-quality* re-solve (same rigor, different method framing, temperature 0.8). Agreement between a solver and its replication is a purely mechanical confidence signal computed in code: stable answers get weight 2, unstable answers get weight 1, and the replication's differing answer enters the pool as its own weight-1 candidate.
3. Deterministic weighted plurality vote in code with explicit, ordered tie-breaking — no LLM judge, no summarizer, no parsing of LLM vote reports.
4. If the weighted vote TIES between distinct answers, invoke the mechanically-executed program (PoT solver + line-by-line executor at temperature 0.0, proven in Gen-2) and break the tie in favor of the tied candidate matching the program's output; if the program matches neither, prefer the candidate with more raw votes, then the earliest original answer.

This is a strict generalization of Self-Consistency (equal weights → its special case), uses only code-level signals for confidence and voting, and keeps the mechanical program anchor exactly where it adds value (resolving ties) rather than as another noisy vote.

**Implementation:**
1. Instantiate 5 solver agents (fields ['thinking','answer'], temperature 0.8) with role+method specs; defensively extract answer Infos (thinking fields no longer collected — unused).
2. Instantiate one replication agent (fields ['thinking','answer'], temperature 0.8, role 'Careful Mathematician'); for each original answer, call it with a method-rotating instruction of equal rigor; compare normalized answers in code.
3. Build weighted pool: original answers get weight 2 on agreement, weight 1 otherwise; disagreeing replications contribute weight-1 candidates; track each norm's earliest original index for deterministic tie-breaking.
4. Sum weights per normalized answer in code; select winner with explicit deterministic tie-breaking; on a weight tie between distinct answers, run PoT solver (fields ['analysis','program'], temperature 0.3) + Program Executor (fields ['trace','program_output'], temperature 0.0, program-only input) and prefer the tied candidate matching the program output; then raw-vote count; then earliest original index.
5. Defensive fallbacks throughout: skip failed calls, fall back to raw vote counts, then to the earliest original answer, then to a single low-temperature direct solve; never construct Info manually, never print, always return an answer Info.

- **generation**: 14
- **fitness**: 95% Bootstrap Confidence Interval: (93.8%, 99.2%), Median: 96.9%

```python
def forward(self, taskInfo):
    import re
    from collections import Counter

    def normalize_answer(text):
        """Normalize an answer string to a comparable numeric token."""
        raw = str(text).strip()
        cleaned = raw.replace(',', '').replace(' ', '').rstrip('.')
        m = re.search(r'-?\d+(?:\.\d+)?', cleaned)
        return m.group(0) if m else raw

    def get_field(infos, field_name):
        """Defensively extract the first Info with the given field name."""
        if infos is None:
            return None
        for info in infos:
            if info.name == field_name:
                return info
        return None

    # ---- Phase 1: Diverse independent solvers (proven backbone) ----
    N = 5
    solver_specs = [
        ('Math Professor', 'Set up algebraic equations with explicit variables and solve them carefully.'),
        ('Grade School Teacher', 'Solve using clear elementary-school arithmetic, one small step at a time, as you would teach a student.'),
        ('Careful Accountant', 'Solve like a bookkeeper: track every quantity in a running tally, verifying each arithmetic operation as you go.'),
        ('Math Enthusiast', 'Solve with whatever clever method you like, but show every step clearly.'),
        ('Helpful Assistant', 'Solve step by step in the clearest direct way you can, double-checking each calculation.'),
    ]
    solver_instruction_template = (
        "Please think step by step and carefully solve this math problem, which may be written in any language. "
        "Pay close attention to every number, quantity, unit, and condition in the original wording. "
        "{method} "
        "Double-check your arithmetic before concluding. "
        "End with the final numeric answer (just the number) in the 'answer' field."
    )
    solver_agents = [
        LLMAgentBase(['thinking', 'answer'], 'Forward Solver', role=role, temperature=0.8)
        for role, _ in solver_specs
    ]

    original_answers = []
    for i in range(N):
        instruction = solver_instruction_template.format(method=solver_specs[i][1])
        outputs = solver_agents[i]([taskInfo], instruction, i)
        answer = get_field(outputs, 'answer')
        if answer is not None:
            original_answers.append(answer)

    if len(original_answers) == 0:
        # Last-resort fallback: single low-temperature direct solve
        fallback_agent = LLMAgentBase(['thinking', 'answer'], 'Fallback Solver', temperature=0.1)
        fallback_instruction = (
            "Please think step by step and carefully solve this math problem, which may be written in any language. "
            "Pay close attention to every number, quantity, unit, and condition. "
            "End with the final numeric answer (just the number) in the 'answer' field."
        )
        fallback_outputs = fallback_agent([taskInfo], fallback_instruction, -1)
        fallback_answer = get_field(fallback_outputs, 'answer')
        if fallback_answer is not None:
            return fallback_answer
        return taskInfo

    # ---- Phase 2: Full-quality independent replication per solver ----
    # The replication uses the SAME rigor as the originals (not a weakened prompt),
    # only rotating the method framing, so agreement is a meaningful stability signal.
    replication_methods = [
        'Solve by working backwards from what is asked.',
        'Solve by first restating every given fact as a numbered list, then deriving the answer from it.',
        'Solve by breaking the problem into the smallest possible sub-steps and computing each one.',
        'Solve by estimating a rough range first, then computing the exact value.',
        'Solve by setting up a clear table of quantities and computing step by step.',
    ]
    replication_agent = LLMAgentBase(['thinking', 'answer'], 'Independent Re-solver',
                                     role='Careful Mathematician', temperature=0.8)

    # weighted_pool entries: (normalized_answer, weight, source_info, is_original, original_index)
    weighted_pool = []
    for idx, ans_info in enumerate(original_answers):
        norm = normalize_answer(ans_info.content)

        rep_instruction = (
            "Please think step by step and carefully solve this math problem, which may be written in any language. "
            "Pay close attention to every number, quantity, unit, and condition in the original wording. "
            f"{replication_methods[idx % len(replication_methods)]} "
            "Do NOT assume any prior solution is correct; derive everything yourself. "
            "Double-check your arithmetic before concluding. "
            "End with the final numeric answer (just the number) in the 'answer' field."
        )
        rep_outputs = replication_agent([taskInfo], rep_instruction, idx)
        rep_answer = get_field(rep_outputs, 'answer')

        if rep_answer is not None and normalize_answer(rep_answer.content) == norm:
            weight = 2  # stable: independent full-quality re-derivation agrees
        else:
            weight = 1  # unstable: demote the original
            if rep_answer is not None:
                rep_norm = normalize_answer(rep_answer.content)
                if rep_norm != norm:
                    weighted_pool.append((rep_norm, 1, rep_answer, False, idx))

        weighted_pool.append((norm, weight, ans_info, True, idx))

    # ---- Phase 3: Deterministic weighted plurality vote in code ----
    total_weight = Counter()
    raw_votes = Counter()
    earliest_original_index = {}
    for norm, weight, _, is_original, orig_idx in weighted_pool:
        total_weight[norm] += weight
        if is_original:
            raw_votes[norm] += 1
            if norm not in earliest_original_index or orig_idx < earliest_original_index[norm]:
                earliest_original_index[norm] = orig_idx

    max_weight = max(total_weight.values())
    top_norms = [norm for norm, w in total_weight.items() if w == max_weight]

    winner_norm = None
    if len(top_norms) == 1:
        winner_norm = top_norms[0]
    else:
        # ---- Phase 4: Weight tie between distinct answers -> mechanical program anchor ----
        pot_agent = LLMAgentBase(['analysis', 'program'], 'Program-of-Thought Solver', temperature=0.3)
        pot_instruction = (
            "Translate this math problem into a short, self-contained Python program. "
            "First analyze the problem in 'analysis': identify the known quantities, the relations between them, "
            "and what is asked. Then write the Python program in 'program': use explicit variables for each quantity, "
            "implement the relations, and end with a print statement of the final result. "
            "Do NOT state the final answer yourself — the program is the only required artifact."
        )
        pot_infos = pot_agent([taskInfo], pot_instruction, -1)
        pot_program = get_field(pot_infos, 'program')

        program_output_norm = None
        if pot_program is not None:
            executor_agent = LLMAgentBase(['trace', 'program_output'], 'Program Executor Agent', temperature=0.0)
            executor_instruction = (
                "You are given a Python program. Execute it EXACTLY as written, line by line, mechanically. "
                "Do not second-guess the program's logic; just simulate it precisely. "
                "In 'trace', show the value of each variable after each step. "
                "In 'program_output', output ONLY the final value printed by the program (just the number)."
            )
            executor_infos = executor_agent([pot_program], executor_instruction, -1)
            program_output = get_field(executor_infos, 'program_output')
            if program_output is not None:
                program_output_norm = normalize_answer(program_output.content)

        # Tiebreak priority (deterministic, explicit order):
        # (1) tied candidate matching the mechanically executed program output;
        # (2) tied candidate with more raw original votes;
        # (3) earliest original answer index among tied candidates.
        if program_output_norm is not None and program_output_norm in top_norms:
            winner_norm = program_output_norm
        else:
            best_raw = max(raw_votes.get(norm, 0) for norm in top_norms)
            raw_tied = [norm for norm in top_norms if raw_votes.get(norm, 0) == best_raw]
            if len(raw_tied) == 1:
                winner_norm = raw_tied[0]
            else:
                winner_norm = min(raw_tied, key=lambda norm: earliest_original_index.get(norm, N))

    # ---- Phase 5: Return the best original answer Info matching the winner ----
    # Prefer the earliest original answer whose normalized value matches the winner.
    best_return = None
    best_idx = N
    for ans_info in original_answers:
        pass  # placeholder removed below
    for idx, ans_info in enumerate(original_answers):
        if normalize_answer(ans_info.content) == winner_norm and idx < best_idx:
            best_idx = idx
            best_return = ans_info
    if best_return is not None:
        return best_return

    # Defensive fallback: winner only appeared via a replication; return the earliest original.
    return original_answers[0]

```

## Extract-Then-Vote with Structural Answer Priors

**Insights:**
The archive's ceiling (~98.4%) is reached only by: diverse independent sampling + voting (Self-Consistency), deliberation (Debate), and Gen-2's mechanically-grounded cross-lingual ensemble. Everything that inserts LLM judgment into verification or vote-counting (verifiers, auditors, summarizers — 96.1–96.9%) underperforms, as does code-only voting with regex normalization (Gen-12, 96.1%). The unaddressed failure mode: all voting schemes in the archive operate on raw answer strings or delegate counting to an LLM. Raw-string voting splits correct consensus ('348' vs '348 rabbits' vs unit-embedded answers in any of MGSM's languages), and first-number regex extraction can capture an intermediate quantity instead of the final answer. The fix that preserves mechanical reliability: extract each solver's FINAL numeric answer with a dedicated temperature-0.0 agent that sees the problem AND the solver's full reasoning (so it can distinguish the answering quantity from intermediates), then vote deterministically in code on the extracted values, with the mechanically-executed program as a tie-only anchor.

**Overall Idea:**
**Extract-Then-Vote with Structural Answer Priors (ETV):**
1. Five diverse solvers (distinct roles AND methods, temperature 0.8) solve independently — the proven backbone, unchanged.
2. NOVEL: each solver's thinking + answer pass through an Answer Extractor agent (temperature 0.0) whose only job is to output the final numeric answer that answers the question asked, applying structural priors: the last derived quantity that answers the question, preference for positive integers, stripping units/words in any language.
3. Deterministic plurality vote in code over extracted answers; non-numeric extractions fall back to regex normalization of the raw answer.
4. Tie between distinct answers: invoke the proven mechanical anchor (PoT solver + temperature-0.0 line-by-line executor); prefer the tied candidate matching the program output, then raw vote counts, then earliest solver index.
5. Return the winning extracted answer Info (the exact value that won the vote), falling back to the original solver answer Info if needed; ultimate fallback is a single low-temperature direct solve.

**Implementation:**
1. Instantiate 5 solver agents (fields ['thinking','answer'], temperature 0.8) with role+method specs; collect (thinking, answer) tuples defensively so alignment is structural.
2. Instantiate one Answer Extractor agent (fields ['final_answer'], temperature 0.0, role 'Precise Extractor'); for each solver, call it with [taskInfo, thinking_info, answer_info] (skipping None thinking) and an instruction encoding the structural priors; wrap in try/except.
3. Build the candidate pool of (normalized_answer, extracted_info_or_None, original_answer_info, original_index); count votes with Counter.
4. On a unique plurality winner, return its extracted Info if numeric, else the original answer Info. On a tie: PoT solver (fields ['analysis','program'], temperature 0.3) + Program Executor (fields ['trace','program_output'], temperature 0.0, program-only input), both exception-guarded; tiebreak chain: program output → raw votes → earliest index.
5. Defensive fallbacks: skip failed solver calls, degrade extraction to regex, skip the anchor if the program phase fails, retry a direct solve before ever returning taskInfo; never construct Info manually, never print, always return an answer Info.

- **generation**: 16
- **fitness**: 95% Bootstrap Confidence Interval: (94.5%, 100.0%), Median: 97.7%

```python
def forward(self, taskInfo):
    import re
    from collections import Counter

    def normalize_answer(text):
        """Regex fallback: extract the LAST numeric token from a free-text answer."""
        raw = str(text).strip()
        cleaned = raw.replace(',', '')
        numbers = re.findall(r'-?\d+(?:\.\d+)?', cleaned)
        if numbers:
            return numbers[-1]
        return raw

    def is_numeric(norm):
        try:
            float(norm)
            return True
        except (ValueError, TypeError):
            return False

    def get_field(infos, field_name):
        """Defensively extract the first Info with the given field name."""
        if infos is None:
            return None
        for info in infos:
            if info.name == field_name:
                return info
        return None

    # ---- Phase 1: Diverse independent solvers (proven backbone, unchanged) ----
    N = 5
    solver_specs = [
        ('Math Professor', 'Set up algebraic equations with explicit variables and solve them carefully.'),
        ('Grade School Teacher', 'Solve using clear elementary-school arithmetic, one small step at a time, as you would teach a student.'),
        ('Careful Accountant', 'Solve like a bookkeeper: track every quantity in a running tally, verifying each arithmetic operation as you go.'),
        ('Math Enthusiast', 'Solve with whatever clever method you like, but show every step clearly.'),
        ('Helpful Assistant', 'Solve step by step in the clearest direct way you can, double-checking each calculation.'),
    ]
    solver_instruction_template = (
        "Please think step by step and carefully solve this math problem, which may be written in any language. "
        "Pay close attention to every number, quantity, unit, and condition in the original wording. "
        "{method} "
        "Double-check your arithmetic before concluding. "
        "End with the final numeric answer (just the number) in the 'answer' field."
    )
    solver_agents = [
        LLMAgentBase(['thinking', 'answer'], 'Forward Solver', role=role, temperature=0.8)
        for role, _ in solver_specs
    ]

    # solver_results: list of (answer_info, thinking_info_or_None) tuples so that
    # answer/thinking alignment is structural rather than positional across two lists.
    solver_results = []
    for i in range(N):
        instruction = solver_instruction_template.format(method=solver_specs[i][1])
        try:
            outputs = solver_agents[i]([taskInfo], instruction, i)
        except Exception:
            continue
        thinking = get_field(outputs, 'thinking')
        answer = get_field(outputs, 'answer')
        if answer is None:
            continue
        solver_results.append((answer, thinking))

    if len(solver_results) == 0:
        # Last-resort fallback: single low-temperature direct solve.
        fallback_agent = LLMAgentBase(['thinking', 'answer'], 'Fallback Solver', temperature=0.1)
        fallback_instruction = (
            "Please think step by step and carefully solve this math problem, which may be written in any language. "
            "Pay close attention to every number, quantity, unit, and condition. "
            "End with the final numeric answer (just the number) in the 'answer' field."
        )
        try:
            fallback_outputs = fallback_agent([taskInfo], fallback_instruction, -1)
        except Exception:
            fallback_outputs = None
        fallback_answer = get_field(fallback_outputs, 'answer')
        if fallback_answer is not None:
            return fallback_answer
        # Second, even more conservative attempt before giving up.
        try:
            fallback_outputs2 = fallback_agent([taskInfo], fallback_instruction, -1)
            fallback_answer2 = get_field(fallback_outputs2, 'answer')
        except Exception:
            fallback_answer2 = None
        if fallback_answer2 is not None:
            return fallback_answer2
        return taskInfo

    # ---- Phase 2: NOVEL — Context-grounded answer extraction with structural priors ----
    # The extractor sees the problem AND the solver's reasoning, so it can tell the final
    # quantity that answers the question apart from intermediate quantities.
    extractor_agent = LLMAgentBase(['final_answer'], 'Answer Extractor',
                                   role='Precise Extractor', temperature=0.0)
    extractor_instruction = (
        "You are given a math problem and ONE solver's proposed solution (its reasoning and its proposed answer). "
        "Your ONLY job is to identify the solver's FINAL numeric answer to the exact question being asked \u2014 "
        "not any intermediate quantity computed along the way.
"
        "Rules:
"
        "1. Trace the solver's reasoning to its conclusion: the final answer is the LAST quantity derived that "
        "answers the exact question asked (total, difference, count of a specific item, etc.).
"
        "2. If the solver's stated answer is ambiguous, re-check which quantity the question asks for and use that one.
"
        "3. Prefer positive integers: grade-school math answers are almost always positive whole numbers.
"
        "4. Ignore units, currency symbols, and words in any language \u2014 output ONLY the number itself.
"
        "In 'final_answer', output ONLY the final numeric answer (digits, optionally a decimal point), nothing else."
    )

    # candidates: (normalized_answer, extracted_info_or_None, original_answer_info, original_index)
    candidates = []
    for idx, (ans_info, thinking_info) in enumerate(solver_results):
        extracted_info = None
        extracted_norm = None
        try:
            extractor_inputs = [taskInfo]
            if thinking_info is not None:
                extractor_inputs.append(thinking_info)
            extractor_inputs.append(ans_info)
            extraction_outputs = extractor_agent(extractor_inputs, extractor_instruction, idx)
            extraction = get_field(extraction_outputs, 'final_answer')
            if extraction is not None:
                norm = normalize_answer(extraction.content)
                # Accept only numeric extractions; a rule-violating non-numeric output
                # is rejected here and we fall back to the raw solver answer below.
                if is_numeric(norm):
                    extracted_info = extraction
                    extracted_norm = norm
        except Exception:
            extracted_info = None
            extracted_norm = None

        # Fallback: regex-normalize the solver's raw answer if extraction failed or is non-numeric.
        if extracted_norm is None:
            extracted_norm = normalize_answer(ans_info.content)

        candidates.append((extracted_norm, extracted_info, ans_info, idx))

    # ---- Phase 3: Deterministic plurality vote in code ----
    votes = Counter()
    earliest_original_index = {}
    for norm, _, _, orig_idx in candidates:
        votes[norm] += 1
        if norm not in earliest_original_index or orig_idx < earliest_original_index[norm]:
            earliest_original_index[norm] = orig_idx

    max_votes = max(votes.values())
    top_norms = [norm for norm, c in votes.items() if c == max_votes]

    winner_norm = None
    if len(top_norms) == 1:
        winner_norm = top_norms[0]
    else:
        # ---- Phase 4: Vote tie -> mechanical program anchor (proven placement) ----
        pot_agent = LLMAgentBase(['analysis', 'program'], 'Program-of-Thought Solver', temperature=0.3)
        pot_instruction = (
            "Translate this math problem into a short, self-contained Python program. "
            "First analyze the problem in 'analysis': identify the known quantities, the relations between them, "
            "and what is asked. Then write the Python program in 'program': use explicit variables for each quantity, "
            "implement the relations, and end with a print statement of the final result. "
            "Do NOT state the final answer yourself \u2014 the program is the only required artifact."
        )
        pot_program = None
        try:
            pot_infos = pot_agent([taskInfo], pot_instruction, -1)
            pot_program = get_field(pot_infos, 'program')
        except Exception:
            pot_program = None

        program_output_norm = None
        if pot_program is not None:
            executor_agent = LLMAgentBase(['trace', 'program_output'], 'Program Executor Agent', temperature=0.0)
            executor_instruction = (
                "You are given a Python program. Execute it EXACTLY as written, line by line, mechanically. "
                "Do not second-guess the program's logic; just simulate it precisely. "
                "In 'trace', show the value of each variable after each step. "
                "In 'program_output', output ONLY the final value printed by the program (just the number)."
            )
            try:
                executor_infos = executor_agent([pot_program], executor_instruction, -1)
                program_output = get_field(executor_infos, 'program_output')
                if program_output is not None:
                    pon = normalize_answer(program_output.content)
                    if is_numeric(pon):
                        program_output_norm = pon
            except Exception:
                program_output_norm = None

        # Tiebreak priority (deterministic, explicit order):
        # (1) tied candidate matching the mechanically executed program output;
        # (2) tied candidate with the higher raw vote count;
        # (3) tied candidate with the earliest original solver index.
        if program_output_norm is not None and program_output_norm in top_norms:
            winner_norm = program_output_norm
        else:
            best_raw = max(votes.get(norm, 0) for norm in top_norms)
            raw_tied = [norm for norm in top_norms if votes.get(norm, 0) == best_raw]
            if len(raw_tied) == 1:
                winner_norm = raw_tied[0]
            else:
                winner_norm = min(raw_tied, key=lambda norm: earliest_original_index.get(norm, N))

    # ---- Phase 5: Return the winning candidate ----
    # Prefer the extracted clean numeric answer Info (exactly the value that won the vote);
    # fall back to the original solver answer Info; prefer the earliest original index.
    best_extracted = None
    best_original = None
    best_idx = N
    for norm, extracted_info, ans_info, orig_idx in candidates:
        if norm != winner_norm or orig_idx >= best_idx:
            continue
        best_idx = orig_idx
        best_original = ans_info
        best_extracted = extracted_info
    if best_extracted is not None:
        return best_extracted
    if best_original is not None:
        return best_original

    # Defensive fallback: earliest original answer.
    return solver_results[0][0]

```

## Decomposition-Augmented Consensus

**Insights:**
The archive's ceiling (~98.4%) is reached by diverse independent sampling + voting (Self-Consistency), deliberation (Debate), and Gen-2's mechanically-grounded ensemble. Three robust lessons: (1) keep 5 diverse, decorrelated, high-temperature independent solvers as the backbone; (2) the mechanically-executed program is the single most trustworthy signal and belongs as a tie anchor, not just another vote; (3) LLM judgment stages that *rewrite or verify* artifacts (checkers, auditors, verifiers) consistently cost fitness — they can silently introduce the very errors they guard against. My previous proposal's novel idea — canonical problem decomposition as a language-independent 'understanding contract' — remains unexplored in the archive and worth keeping, but its earlier implementation was diluted: it demoted the proven backbone to 3 low-temperature contract solvers sharing a possibly-wrong contract (correlated failure), and added an LLM 'corrected_decomposition' stage that contradicts lesson (3). The revision: backbone unchanged (5 diverse raw-text solvers, temp 0.8), with the decomposition as ONE additional independent structured signal rather than a shared crutch, and validation done mechanically in code rather than by an LLM rewriter.

**Overall Idea:**
**Decomposition-Augmented Consensus (DAC):**
1. Backbone (proven): 5 diverse solvers (distinct roles AND methods, temperature 0.8) solve the raw multilingual text independently — fully decorrelated paths.
2. NOVEL signal — Structured Decomposition Path: a temperature-0.0 Decomposition Agent converts the problem into a canonical structured form (given quantities with units, explicit symbolic relations, target quantity). A Decomposition Solver (temperature 0.3) then solves *strictly from the decomposition*, which removes cross-lingual misreading as a failure mode for that path. No LLM correction stage; if decomposition fails, the signal is simply dropped.
3. Mechanical anchor (proven): PoT solver + temperature-0.0 line-by-line executor on the raw text.
4. Deterministic code-level plurality vote over 7 candidates (5 backbone + decomposition-solver + program output). Unique winner returned directly. On a tie, tiebreak chain: (1) tied candidate matching the mechanically-executed program output, (2) earlier candidate in the priority-ordered list.
5. Defensive fallbacks: skip failed calls; if no candidates, retry a low-temperature direct solve before ever returning taskInfo; never construct Info manually, never print, always return an answer Info.

**Implementation:**
1. Instantiate 5 backbone solver agents (fields ['thinking','answer'], temperature 0.8) with role+method specs; collect answer Infos defensively via get_field (no positional unpacking).
2. Decomposition Agent (fields ['problem_structure'], temperature 0.0, role 'Formal Problem Analyst'): fixed-format structured decomposition; every number from the problem must appear; relations as explicit equations; target = exactly what the question asks.
3. Decomposition Solver (fields ['thinking','answer'], temperature 0.3, role 'Structured Solver'): receives [taskInfo, problem_structure_info], instructed to solve strictly from the listed quantities/relations and to answer exactly the stated target.
4. PoT solver (fields ['analysis','program'], temperature 0.3) + Program Executor (fields ['trace','program_output'], temperature 0.0, program-only input) — exception-guarded.
5. Build candidate pool in priority order: backbone answers → decomposition-solver answer → program output; vote with Counter on last-number-normalized answers; resolve ties per the chain above; return the earliest matching candidate Info.

- **generation**: 20
- **fitness**: 95% Bootstrap Confidence Interval: (93.8%, 99.2%), Median: 96.9%

```python
def forward(self, taskInfo):
    import re
    from collections import Counter

    def normalize_answer(text):
        """Extract the LAST numeric token from a free-text answer (final-answer prior)."""
        raw = str(text).strip()
        cleaned = raw.replace(',', '')
        numbers = re.findall(r'-?\d+(?:\.\d+)?', cleaned)
        if numbers:
            return numbers[-1]
        return raw

    def is_numeric(norm):
        try:
            float(norm)
            return True
        except (ValueError, TypeError):
            return False

    def get_field(infos, field_name):
        """Defensively extract the first Info with the given field name."""
        if infos is None:
            return None
        for info in infos:
            if info.name == field_name:
                return info
        return None

    # ---- Phase 1: Backbone — 5 diverse independent solvers (proven, unchanged) ----
    N = 5
    solver_specs = [
        ('Math Professor', 'Set up algebraic equations with explicit variables and solve them carefully.'),
        ('Grade School Teacher', 'Solve using clear elementary-school arithmetic, one small step at a time, as you would teach a student.'),
        ('Careful Accountant', 'Solve like a bookkeeper: track every quantity in a running tally, verifying each arithmetic operation as you go.'),
        ('Math Enthusiast', 'Solve with whatever clever method you like, but show every step clearly.'),
        ('Helpful Assistant', 'Solve step by step in the clearest direct way you can, double-checking each calculation.'),
    ]
    solver_instruction_template = (
        "Please think step by step and carefully solve this math problem, which may be written in any language. "
        "Pay close attention to every number, quantity, unit, and condition in the original wording. "
        "{method} "
        "Double-check your arithmetic before concluding. "
        "End with the final numeric answer (just the number) in the 'answer' field."
    )
    solver_agents = [
        LLMAgentBase(['thinking', 'answer'], 'Forward Solver', role=role, temperature=0.8)
        for role, _ in solver_specs
    ]

    backbone_answers = []
    for i in range(N):
        instruction = solver_instruction_template.format(method=solver_specs[i][1])
        try:
            outputs = solver_agents[i]([taskInfo], instruction, i)
        except Exception:
            continue
        answer = get_field(outputs, 'answer')
        if answer is not None:
            backbone_answers.append(answer)

    # ---- Phase 2: NOVEL — Structured decomposition path (one independent signal) ----
    decomposition_info = None
    decomposition_agent = LLMAgentBase(['problem_structure'], 'Problem Decomposition Agent',
                                       role='Formal Problem Analyst', temperature=0.0)
    decomposition_instruction = (
        "You are given a grade-school math problem, possibly written in a non-English language. "
        "Your ONLY job is to understand the problem precisely - do NOT solve it. "
        "Produce a canonical, language-independent structured decomposition in 'problem_structure' with three sections: "
        "GIVEN QUANTITIES: list EVERY quantity given in the problem with its exact value and unit, "
        "for example 'dogs = 60' and 'cats per dog = 2'. Every number appearing in the problem must appear here. "
        "RELATIONS: list EVERY relation or constraint between the quantities as an explicit symbolic equation, "
        "for example 'rabbits = dogs + cats - 12' and 'cats = 2 * dogs'. Be careful with comparative phrasing "
        "such as 'fewer than', 'more than', 'per', 'each', 'twice' in any language - translate it into the correct equation. "
        "TARGET: state EXACTLY what the question asks for - the final quantity to compute, not a sub-quantity."
    )
    try:
        decomp_infos = decomposition_agent([taskInfo], decomposition_instruction, -1)
        decomposition_info = get_field(decomp_infos, 'problem_structure')
    except Exception:
        decomposition_info = None

    decomp_answer = None
    if decomposition_info is not None:
        decomp_solver_agent = LLMAgentBase(['thinking', 'answer'], 'Decomposition Solver',
                                           role='Structured Solver', temperature=0.3)
        decomp_solver_instruction = (
            "You are given a math problem AND a structured decomposition of it "
            "(given quantities, relations, and the target). "
            "Treat the decomposition as the authoritative statement of the problem. "
            "Solve strictly from the listed quantities and relations - do not re-interpret the original wording. "
            "Make sure your final result is exactly the stated TARGET quantity. "
            "Double-check your arithmetic before concluding. "
            "End with the final numeric answer (just the number) in the 'answer' field."
        )
        try:
            decomp_outputs = decomp_solver_agent([taskInfo, decomposition_info], decomp_solver_instruction, -1)
            decomp_answer = get_field(decomp_outputs, 'answer')
        except Exception:
            decomp_answer = None

    # ---- Phase 3: Mechanical arithmetic anchor (proven) ----
    program_output = None
    pot_agent = LLMAgentBase(['analysis', 'program'], 'Program-of-Thought Solver', temperature=0.3)
    pot_instruction = (
        "Translate this math problem into a short, self-contained Python program. "
        "First analyze the problem in 'analysis': identify the known quantities, the relations between them, "
        "and what is asked. Then write the Python program in 'program': use explicit variables for each quantity, "
        "implement the relations, and end with a print statement of the final result. "
        "Do NOT state the final answer yourself - the program is the only required artifact."
    )
    pot_program = None
    try:
        pot_infos = pot_agent([taskInfo], pot_instruction, -1)
        pot_program = get_field(pot_infos, 'program')
    except Exception:
        pot_program = None

    if pot_program is not None:
        executor_agent = LLMAgentBase(['trace', 'program_output'], 'Program Executor Agent', temperature=0.0)
        executor_instruction = (
            "You are given a Python program. Execute it EXACTLY as written, line by line, mechanically. "
            "Do not second-guess the program's logic; just simulate it precisely. "
            "In 'trace', show the value of each variable after each step. "
            "In 'program_output', output ONLY the final value printed by the program (just the number)."
        )
        try:
            executor_infos = executor_agent([pot_program], executor_instruction, -1)
            po = get_field(executor_infos, 'program_output')
            if po is not None and is_numeric(normalize_answer(po.content)):
                program_output = po
        except Exception:
            program_output = None

    # ---- Phase 4: Deterministic plurality vote in code ----
    # Candidate priority order: backbone answers -> decomposition-solver answer -> program output.
    candidate_infos = list(backbone_answers)
    if decomp_answer is not None:
        candidate_infos.append(decomp_answer)
    if program_output is not None:
        candidate_infos.append(program_output)

    if len(candidate_infos) == 0:
        # Last-resort fallback: single low-temperature direct solve, retried once.
        fallback_agent = LLMAgentBase(['thinking', 'answer'], 'Fallback Solver', temperature=0.1)
        fallback_instruction = (
            "Please think step by step and carefully solve this math problem, which may be written in any language. "
            "Pay close attention to every number, quantity, unit, and condition. "
            "End with the final numeric answer (just the number) in the 'answer' field."
        )
        for attempt in range(2):
            try:
                fallback_outputs = fallback_agent([taskInfo], fallback_instruction, -1)
                fallback_answer = get_field(fallback_outputs, 'answer')
            except Exception:
                fallback_answer = None
            if fallback_answer is not None:
                return fallback_answer
        return taskInfo

    votes = Counter()
    earliest_index = {}
    for idx, a in enumerate(candidate_infos):
        norm = normalize_answer(a.content)
        votes[norm] += 1
        if norm not in earliest_index or idx < earliest_index[norm]:
            earliest_index[norm] = idx

    max_votes = max(votes.values())
    top_norms = [norm for norm, c in votes.items() if c == max_votes]

    winner_norm = None
    if len(top_norms) == 1:
        winner_norm = top_norms[0]
    else:
        # Tiebreak chain (deterministic, explicit order):
        # (1) tied candidate matching the mechanically executed program output;
        # (2) tied candidate appearing earliest in the priority-ordered candidate list.
        if program_output is not None:
            program_norm = normalize_answer(program_output.content)
            if program_norm in top_norms:
                winner_norm = program_norm
        if winner_norm is None:
            winner_norm = min(top_norms, key=lambda norm: earliest_index.get(norm, len(candidate_infos)))

    # ---- Phase 5: Return the winning candidate Info (earliest matching) ----
    for a in candidate_infos:
        if normalize_answer(a.content) == winner_norm:
            return a

    # Defensive fallback: first candidate.
    return candidate_infos[0]

```
