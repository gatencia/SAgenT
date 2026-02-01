this is just a scrap of ideas I could potentially implement in the future. 


####################################################################################
Several Options in hwo to make the agent actually learn important things. 

1. Using dreamcoder and a learning library to learn abstractions onhow to solve these problems. I would use LILO (Library Induction from Language Observations). 
2. We could ideally get another LLM to suggest the closest problems using the learning library

here is a reseaerch querry I got to implement such a db:

Implementation Roadmap: 
Building the Encoding Inspiration LibraryThe following plan describes the transition from raw natural language (NL) data to a structural memory that enables an agent to solve expert-level encoding tasks by analogy.

Phase 1:
Seed Acquisition (The "Reading" Phase)First, you populate your agent's initial experience using SATBench, a Stanford-developed benchmark of 2,100 puzzles specifically designed for this purpose.

1. Extract Mappings: Download the SATBench dataset. For each instance, extract the three critical components: the Story Background (context), the Narrative Conditions (constraints), and the Variable Mapping (the ground-truth link between entities and Boolean variables).

2. Generate Logic Scripts: Use your LLM to translate these NL components into high-level Python logic scripts (using PySAT or Z3). Instead of raw CNF, the model should write code like g.add_clause().

3. Correctness Verification: Execute these scripts and compare the resulting truth tables or satisfiability labels against SATBench's ground truth to ensure the LLM correctly captured the logic.

Phase 2: 

Graph Canonicalization (Removing "Surface" Details)To enable analogical reasoning, the agent must ignore whether a problem is about "cats" or "musicians" and focus only on the "shape" of the constraints.

1. LCG Construction: Convert every successful logic script into a Literal-Clause Graph (LCG). This is a bipartite graph where one set of nodes represents variables ($x_i, \neg x_i$) and the other represents the clauses (rules).

2. Structural Embedding: Pass these graphs through a Graph Neural Network (GNN) encoder (such as SAT-GATv2). This "squashes" the complex graph topology into a fixed-length numerical vector (an embedding) that captures the relational essence of the problem.

3. Canonical Indexing: Store these embeddings in a Vector Database (e.g., ChromaDB or FAISS). Each vector serves as a "structural fingerprint" that is linked to the Python code that generated it.

Phase 3: Library Learning (The "Sleep" Cycle)

In this phase, the agent identifies recurring patterns across the 2,100 problems to "invent" new encoding tools.

1. S-Expression Conversion: Convert your logic scripts into lambda calculus (Lisp-like format) using de Bruijn indices.

2. Symbolic Compression (Stitch): Run the Stitch algorithm on your entire corpus of programs. Stitch identifies which logical sub-structures repeat (e.g., the specific way a "pigeonhole" or "XOR-chain" is written) and abstracts them into new, reusable functions called Gadgets.

3. Auto-Documentation (LLM-in-the-loop): Pass these cryptic gadgets (e.g., fn_12) to an LLM with examples of how they were used. The LLM labels them with names like "Cyclic Dependency Gadget" and provides a natural language description.


Phase 4: Analogical Solving (The "Expert" Workflow)Now, when you provide a new, unsolved NL problem, the agent uses its library for inspiration:

1. Retrieve by Analogy: The agent extracts a high-level "constraint graph" from the new NL problem and embeds it. It queries the Vector DB for the Top-K most structurally similar problems from Phase 

2. Strategy Proposal: The agent retrieves the Gadgets used in those similar problems. It prompts the LLM: "This new logistics problem is structurally 89% similar to a 'Job Shop' problem in our library. Use the 'Symmetry Breaking' gadget from that case.".

3.Test-Time Optimization: The agent generates multiple candidate encodings using different gadgets. It runs a SAT solver in a tight loop, using the number of conflicts or runtime as a reward signal to find the "Best" representation for this unique instance.

Phase 5: 
Deployment in DockerTo keep this system stable and portable:

1. Agent Environment: Run the LLM agent, the Kissat solver, and the Stitch engine inside a Docker container.

2. Persistent Storage: Map the Vector Database and the JSON Library (the gadgets) to a Docker Volume so the agent's "experience" persists even if the container is restarted.

Summary of the Workflow:By starting with SATBench, you provide the agent with its first 2,100 "memories." By converting these memories into Literal-Clause Graphs, you strip away the story and leave the mathematical skeleton. Library Learning then turns those skeletons into a toolkit of "Gadgets" that the agent can retrieve by analogy to solve expert-level problems it has never seen before.





####################################################################################
Type of graphs and logs I should consider getting
1. IR influence on reasoning:
    - Number of constraints changed over number of steps
        --> this could give us clues on how an IR influences the exploration of the space of solutions
    - Number of denials for each stage of the problem / amount of time spent on each stage of the problem
        --> This oculd give us hints on what specific part of the solution discovery an IR is best at

2. Comparison to an LLM without help of the SATManager. 
    - Amount of time until result 
    - Number of trials until result over differnt problems. 



####################################################################################

Analogical reasoning as inspiration for Solutions.  


####################################################################################
Figuring out when to use what IR. Similar to when an LLM decides when to do deepresearch, thinking or a flash answer, we should figure out how to structuer the reasoning to use the best IR for the current step of the problem solving process. 